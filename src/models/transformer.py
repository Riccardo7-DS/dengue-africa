import torch
import torch.nn as nn
import timm

# -------------------------
# High-res branch (patch-based Swin Transformer)
# -------------------------

class HighResBranch(nn.Module):
    def __init__(self, in_ch=3, out_dim=128, patch_size=224, pretrained=True):
        super().__init__()
        self.patch_size = patch_size

        # Load Swin-T pretrained on ImageNet-1k as a feature extractor
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in1k',
            pretrained=pretrained,
            in_chans=in_ch,
            num_classes=0  # remove classifier head
        )
        self.feat_dim = self.swin.num_features

        # Project Swin features to desired output dimension
        self.proj = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x):
        # x: [B, C, H, W] e.g., [4, 3, 1024, 1024]
        B, C, H, W = x.shape
        p = self.patch_size

        # Compute number of patches along height and width
        num_patches_H = H // p
        num_patches_W = W // p

        # Extract non-overlapping patches using unfold
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B, C, num_H, num_W, p, p]
        patches = patches.contiguous().view(B, C, -1, p, p)  # [B, C, num_patches, p, p]
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, C, p, p)  # [B*num_patches, C, p, p]

        # Forward each patch through Swin
        patch_feats = self.swin(patches)  # [B*num_patches, feat_dim]

        # Reshape back to batch
        patch_feats = patch_feats.view(B, num_patches_H * num_patches_W, -1)  # [B, num_patches, feat_dim]

        # Aggregate patch features (mean pooling)
        image_feat = patch_feats.mean(dim=1)  # [B, feat_dim]

        # Project to desired output dimension
        out = self.proj(image_feat)  # [B, out_dim]
        return out

# -------------------------
# Medium-res branch (CNN + patch pooling)
# -------------------------
class MediumResBranch(nn.Module):
    def __init__(self, in_ch=2, out_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        feat = self.cnn(x)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)  # (B, 128)
        feat = self.fc(feat)                             # (B, out_dim)
        return feat
    
class MediumResBranchAttention(nn.Module):
    def __init__(self, in_ch=8, out_dim=128, hidden_dim=128, num_heads=4):
        super().__init__()
        # Spatial CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        # Temporal attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        # Final projection
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # Flatten batch and time to process with CNN
        x_2d = x.view(B*T, C, H, W)          # [B*T, C, H, W]
        feat2d = self.cnn(x_2d)              # [B*T, hidden_dim, H, W]

        # Global average pooling over spatial dims
        feat2d = feat2d.mean(dim=[2,3])      # [B*T, hidden_dim]

        # Restore batch and time dims
        feat2d = feat2d.view(B, T, -1)       # [B, T, hidden_dim]

        # Temporal attention
        attn_out, _ = self.attn(feat2d, feat2d, feat2d)  # [B, T, hidden_dim]

        # Pool over time
        feat = attn_out.mean(dim=1)          # [B, hidden_dim]

        # Project to output dimension
        out = self.fc(feat)                  # [B, out_dim]
        return out

# -------------------------
# Coarse/static branch (linear embedding)
# -------------------------
class StaticBranch(nn.Module):
    def __init__(self, in_ch=1, out_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))   # pool spatial map to 1x1
        self.fc = nn.Sequential(
            nn.Flatten(),                          # [B, C]
            nn.Linear(in_ch, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pool(x)   # [B, C, 1, 1]
        out = self.fc(x)   # [B, out_dim]
        return out

# -------------------------
# Fusion + Prediction
# -------------------------
class DenguePredictor(nn.Module):
    def __init__(self,
                 high_in_ch=3,
                 med_in_ch=2,
                 static_in_ch=1,
                 high_out=128,
                 med_out=128,
                 static_out=128,
                 hidden_dim=256,
                 output_dim=1,
                 output_size=(86, 86),
                 output_channels=1,
                 decoder_channels=64):
        super().__init__()
        self.high_branch = HighResBranch(out_dim=high_out, in_ch=high_in_ch)
        self.med_branch = MediumResBranchAttention(out_dim=med_out, in_ch=med_in_ch)
        self.static_branch = StaticBranch(out_dim=static_out, in_ch=static_in_ch)

        # Output configuration
        self.output_size = output_size
        self.output_channels = output_channels
        self.decoder_channels = decoder_channels

        # Fusion head (MLP) that projects to a small spatial feature map,
        # which is then decoded by convolutional layers to produce the final image.
        total_in = high_out + med_out + static_out
        H, W = output_size
        proj_pixels = decoder_channels * H * W
        self.fc = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_pixels),
            nn.ReLU()
        )

        # Simple conv-based decoder: refines the projected feature-map to the final output
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_channels, output_channels, kernel_size=1)
        )

    def forward(self, x_high, x_med, x_static):
        f_high = self.high_branch(x_high)
        f_med = self.med_branch(x_med)
        f_static = self.static_branch(x_static)

        # Concatenate embeddings
        fusion = torch.cat([f_high, f_med, f_static], dim=1)

        # Predict dengue risk / cases: project to spatial feature-map then decode
        out = self.fc(fusion)

        # Reshape to feature-map: (B, decoder_channels, H, W)
        B = out.shape[0]
        H, W = self.output_size
        out = out.view(B, self.decoder_channels, H, W)

        # Decode to final image
        out = self.decoder(out)
        return out