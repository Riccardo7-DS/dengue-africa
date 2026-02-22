import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# -------------------------
# High-res branch (patch-based Swin Transformer)
# -------------------------
class HighResBranch(nn.Module):
    def __init__(self, in_ch=3, out_dim=128, patch_size=256, pretrained=True):
        super().__init__()
        self.patch_size = patch_size

        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224.ms_in1k',
            pretrained=pretrained,
            in_chans=in_ch,
            num_classes=0,
            img_size=patch_size  # 256 divides 1024 cleanly → 4x4 = 16 patches
        )
        self.swin.set_grad_checkpointing(True) # save memory by recomputing activations during backward pass
        self.feat_dim = self.swin.num_features  # 768 for swin_tiny

        # Attention pooling over patches instead of mean pooling
        self.patch_attn = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.proj = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        num_H = H // p
        num_W = W // p

        # Extract non-overlapping patches
        patches = x.unfold(2, p, p).unfold(3, p, p)          # [B, C, num_H, num_W, p, p]
        patches = patches.contiguous().view(B, C, -1, p, p)  # [B, C, N, p, p]
        patches = patches.permute(0, 2, 1, 3, 4).reshape(-1, C, p, p)  # [B*N, C, p, p]

        # Forward through Swin
        patch_feats = self.swin(patches)                      # [B*N, feat_dim]
        N = num_H * num_W
        patch_feats = patch_feats.view(B, N, self.feat_dim)  # [B, N, feat_dim]

        # Attention pooling over patches (learns which patches matter more)
        attn_w = self.patch_attn(patch_feats)                 # [B, N, 1]
        attn_w = torch.softmax(attn_w, dim=1)                 # [B, N, 1]
        image_feat = (attn_w * patch_feats).sum(dim=1)        # [B, feat_dim]

        return self.proj(image_feat)                          # [B, out_dim]


class MediumResBranchAttention(nn.Module):
    def __init__(self, in_ch=2, out_dim=128, hidden_dim=128, num_heads=4):
        super().__init__()

        # Spatial CNN encoder — extracts local spatial features per timestep
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # Spatial attention: learns which spatial locations matter
        # Applied per timestep before temporal aggregation
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),  # [B*T, 1, H, W]
            nn.Sigmoid()
        )

        self.pre_norm = nn.LayerNorm(hidden_dim)

        # Temporal attention across timesteps
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # CNN per timestep
        x_2d = x.view(B * T, C, H, W)
        feat2d = self.cnn(x_2d)                        # [B*T, hidden_dim, H, W]

        # Spatial attention: weight spatial locations before pooling
        spatial_w = self.spatial_attn(feat2d)          # [B*T, 1, H, W]
        feat2d = (feat2d * spatial_w).sum(dim=[2, 3])  # [B*T, hidden_dim] — weighted spatial pool

        # Restore time dimension
        feat2d = feat2d.view(B, T, -1)                 # [B, T, hidden_dim]
        
        residual = feat2d
        feat2d = self.pre_norm(feat2d)                # LayerNorm before attention
        # Temporal self-attention with residual
        attn_out, _ = self.temporal_attn(feat2d, feat2d, feat2d)  # [B, T, hidden_dim]
        feat2d = residual + attn_out          # residual connection

        feat2d = self.post_norm(feat2d)
        # Pool over time
        feat = feat2d.mean(dim=1)                      # [B, hidden_dim]

        return self.fc(feat)                           # [B, out_dim]


class StaticBranch(nn.Module):
    def __init__(self, in_ch=1, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)                         # [B, out_dim]


class DenguePredictor(nn.Module):
    def __init__(self,
                 high_in_ch=3,
                 med_in_ch=2,
                 static_in_ch=1,
                 high_out=128,
                 med_out=128,
                 static_out=128,
                 hidden_dim=256,
                 output_size=(86, 86),
                 output_channels=1,
                 decoder_channels=64):
        super().__init__()
        self.high_branch = HighResBranch(out_dim=high_out, in_ch=high_in_ch)
        self.med_branch = MediumResBranchAttention(out_dim=med_out, in_ch=med_in_ch)
        self.static_branch = StaticBranch(out_dim=static_out, in_ch=static_in_ch)

        self.output_size = output_size
        self.decoder_channels = decoder_channels

        total_in = high_out + med_out + static_out

        # Project to small spatial grid then upsample — avoids giant linear layer
        self.fc = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decoder_channels * 8 * 8),  # 8x8 bottleneck
            nn.ReLU()
        )

        # Upsample + refine to output_size
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(),
            nn.Conv2d(decoder_channels, output_channels, 1)
        )

    def forward(self, x_high, x_med, x_static):
        f_high = self.high_branch(x_high)
        f_med = self.med_branch(x_med)
        f_static = self.static_branch(x_static)

        fusion = torch.cat([f_high, f_med, f_static], dim=1)  # [B, total_in]

        out = self.fc(fusion)                                  # [B, decoder_channels * 64]

        B = out.shape[0]
        out = out.view(B, self.decoder_channels, 8, 8)         # [B, decoder_channels, 8, 8]

        # Upsample to target output size
        out = F.interpolate(out, size=self.output_size, mode='bilinear', align_corners=False)

        return self.decoder(out)                               # [B, 1, 86, 86]

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