import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2

# -------------------------------
# Define Pyramid Pooling Module (with GroupNorm)
# -------------------------------
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.pool_layers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=in_channels // 4),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        total_channels = in_channels + len(pool_sizes) * (in_channels // 4)
        self.conv = nn.Conv2d(total_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        pooled_features = [x]
        for layer in self.pool_layers:
            pooled = layer(x)
            pooled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
            pooled_features.append(pooled)
        x = torch.cat(pooled_features, dim=1)
        x = self.conv(x)
        return x

# -------------------------------
# Define UPerNet Decoder (With Dropout)
# -------------------------------
class UPerNetDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1, dropout_rate=0.1):
        super().__init__()
        self.ppm = PyramidPoolingModule(encoder_channels[-1])
        self.lateral_conv2 = nn.Conv2d(encoder_channels[2], encoder_channels[-1], kernel_size=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], encoder_channels[2], kernel_size=1),
            nn.Dropout2d(p=dropout_rate)
        )
        self.lateral_conv1 = nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channels[2], encoder_channels[1], kernel_size=1),
            nn.Dropout2d(p=dropout_rate)
        )
        self.lateral_conv0 = nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_channels[1], encoder_channels[0], kernel_size=1),
            nn.Dropout2d(p=dropout_rate)
        )
        self.segmentation_head = nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, features):
        f0, f1, f2, f3 = features
        x3 = self.ppm(f3)
        x3_up = F.interpolate(x3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        fuse2 = x3_up + self.lateral_conv2(f2)
        fuse2 = self.conv3(fuse2)
        fuse2_up = F.interpolate(fuse2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        fuse1 = fuse2_up + self.lateral_conv1(f1)
        fuse1 = self.conv2(fuse1)
        fuse1_up = F.interpolate(fuse1, size=f0.shape[2:], mode="bilinear", align_corners=False)
        fuse0 = fuse1_up + self.lateral_conv0(f0)
        fuse0 = self.conv1(fuse0)
        x_out = F.interpolate(fuse0, size=(224, 224), mode="bilinear", align_corners=False)
        output = self.segmentation_head(x_out)
        return output

# -------------------------------
# Define Swin-Tiny UPerNet Model
# -------------------------------
class SwinTinyUPerNet(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.1):
        super().__init__()
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
            pretrained=True,
            features_only=True
        )
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = UPerNetDecoder(encoder_channels, num_classes, dropout_rate=dropout_rate)

    def forward(self, x):
        features = self.encoder(x)
        features = [f.permute(0, 3, 1, 2) if f.dim() == 4 else f for f in features]
        output = self.decoder(features)
        return F.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)

# -------------------------------
# Load the Model
# -------------------------------
def load_model():
    model = SwinTinyUPerNet(num_classes=1)
    model.load_state_dict(torch.load("best_swin_upernet_main.pth", map_location=torch.device("cpu")), strict=False)
    model.eval()
    return model

# -------------------------------
# Enable Dropout at Inference Time
# -------------------------------
def enable_dropout(m):
    if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
        m.train()

# -------------------------------
# Perform Inference with MC Dropout
# -------------------------------
def predict_with_uncertainty(image_tensor, num_samples=10):
    model = load_model()
    model.apply(enable_dropout)
    preds_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            preds = torch.sigmoid(model(image_tensor))
            preds_list.append(preds)

    preds_array = torch.stack(preds_list, dim=0)
    preds_mean = preds_array.mean(dim=0).squeeze().cpu().numpy()
    preds_uncertainty = preds_array.std(dim=0).squeeze().cpu().numpy()

    # Normalize uncertainty map
    preds_uncertainty = (preds_uncertainty - preds_uncertainty.min()) / (preds_uncertainty.max() - preds_uncertainty.min() + 1e-8)
    return preds_mean, preds_uncertainty