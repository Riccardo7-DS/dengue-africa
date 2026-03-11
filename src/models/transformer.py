import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


import torch
import torch.nn as nn
import timm


class TiTokBottleneck(nn.Module):
    """
    Compresses N patch tokens → K latent tokens via cross-attention.
    Mimics TiTok's 1D tokenization without requiring a VQ-VAE codebook.
    K << N achieves the compact 1D representation.
    """
    def __init__(self, feat_dim: int, num_latent_tokens: int = 32, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.latent_tokens = nn.Parameter(torch.randn(1, num_latent_tokens, feat_dim))

        # Cross-attention: latent queries attend over patch keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_q = nn.LayerNorm(feat_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)

        # Self-attention among latent tokens (lets them communicate)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # pre-norm, more stable
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N, feat_dim]  — dense patch features from backbone
        Returns:
            latent: [B, K, feat_dim]  — compact 1D token sequence
        """
        B = patch_tokens.size(0)

        # Expand learned queries to batch
        queries = self.latent_tokens.expand(B, -1, -1)          # [B, K, feat_dim]

        # Cross-attend: K latent tokens query N patch tokens
        latent, _ = self.cross_attn(
            query=self.norm_q(queries),
            key=self.norm_kv(patch_tokens),
            value=self.norm_kv(patch_tokens)
        )                                                         # [B, K, feat_dim]

        # Self-attention: latent tokens refine each other
        latent = self.self_attn(latent)                          # [B, K, feat_dim]

        return latent


class HighResBranchTiTok(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_dim: int = 128,
        patch_size: int = 256,
        num_latent_tokens: int = 32,   # K — the "1D sequence length", tune this (16–64)
        pretrained: bool = True,
        backbone: str = 'vit_base_patch16_224.mae'
    ):
        super().__init__()
        self.patch_size = patch_size

        # --- Backbone (patch feature extractor) ---
        # forward_features() returns [B, N, feat_dim] token grid, not a pooled vector
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_ch,
            num_classes=0,
            img_size=patch_size,
            global_pool=''          # IMPORTANT: disable pooling, keep all patch tokens
        )
        self.encoder.set_grad_checkpointing(True)
        self.feat_dim = self.encoder.num_features   # e.g. 768 for ViT-B

        # --- TiTok-style bottleneck ---
        self.titok = TiTokBottleneck(
            feat_dim=self.feat_dim,
            num_latent_tokens=num_latent_tokens
        )

        # --- Attention pooling over latent tokens → single vector ---
        self.pool_attn = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.proj = nn.Linear(self.feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        num_H, num_W = H // p, W // p
        N_patches = num_H * num_W

        # --- 1. Tile large image into non-overlapping spatial patches ---
        patches = x.unfold(2, p, p).unfold(3, p, p)             # [B, C, nH, nW, p, p]
        patches = patches.contiguous().view(B, C, -1, p, p)     # [B, C, N, p, p]
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(-1, C, p, p)                  # [B*N, C, p, p]

        # --- 2. Extract patch tokens via backbone (no pooling) ---
        token_grid = self.encoder.forward_features(patches)      # [B*N, T, feat_dim]

        # Strip CLS token if present (ViT prepends one)
        if token_grid.shape[1] > 1:
            token_grid = token_grid[:, 1:, :]                   # [B*N, T-1, feat_dim]

        T = token_grid.shape[1]

        # --- 3. TiTok bottleneck: T dense tokens → K latent tokens ---
        latent = self.titok(token_grid)                          # [B*N, K, feat_dim]

        # --- 4. Merge spatial tiles: concatenate latent sequences ---
        K = latent.shape[1]
        latent = latent.view(B, N_patches, K, self.feat_dim)
        latent = latent.reshape(B, N_patches * K, self.feat_dim)   # [B, N*K, feat_dim]

        # --- 5. Attention pooling over full latent sequence → image vector ---
        attn_w = self.pool_attn(latent)                          # [B, N*K, 1]
        attn_w = torch.softmax(attn_w, dim=1)
        image_feat = (attn_w * latent).sum(dim=1)               # [B, feat_dim]

        return self.proj(image_feat)                             # [B, out_dim]

# -------------------------
# High-res branch (patch-based Swin Transformer)
# -------------------------
class HighResBranch(nn.Module):
    def __init__(self, in_ch=3, out_dim=128, patch_size=256, pretrained=True, swin_model='swin_tiny_patch4_window7_224.ms_in1k'):
        super().__init__()
        self.patch_size = patch_size

        self.encoder = timm.create_model(
            swin_model,
            pretrained=pretrained,
            in_chans=in_ch,
            num_classes=0,
            img_size=patch_size  # 256 divides 1024 cleanly → 4x4 = 16 patches
        )
        self.encoder.set_grad_checkpointing(True) # save memory by recomputing activations during backward pass
        self.feat_dim = self.encoder.num_features  # 

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
        patch_feats = self.encoder(patches)                      # [B*N, feat_dim]
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
    
class ZoneEmbeddingBlock(nn.Module):
    def __init__(self, num_zones, embed_dim):
        super().__init__()
        # +1 to reserve index 0 as padding token for out-of-zone pixels
        self.embedding = nn.Embedding(num_zones + 1, embed_dim, padding_idx=0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        # x: [B, 1, 1, H, W],[B, 1, H, W] or [B, H, W]
        if x.dim() == 5:
            x = x.squeeze(1).squeeze(1).long() # [B, H, W]
        
        if x.dim() == 4:
            x = x.squeeze(1)           # [B, H, W]
        
        x = x.long()
        x = (x + 1).clamp(0, self.embedding.num_embeddings - 1)  # -1 → 0 (padding)

        out = self.embedding(x)        # [B, H, W, embed_dim]
        out = out.permute(0, 3, 1, 2)  # [B, embed_dim, H, W]
        out = self.pool(out)           # [B, embed_dim, 1, 1]
        return self.flatten(out)       # [B, embed_dim]


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

# -------------------------
# Fusion + Prediction
# -------------------------
class DenguePredictor(nn.Module):
    def __init__(self,
                 swin_model="swin_tiny_patch4_window7_224.ms_in1k",
                 high_in_ch=3,
                 med_in_ch=2,
                 static_in_ch=1,
                 high_out=128,
                 med_out=128,
                 static_out=128,
                 hidden_dim=256,
                 output_size=(86, 86),
                 output_channels=1,
                 decoder_channels=64,
                 num_zones=8000,
                 # --- TiTok options ---
                 use_titok=False,
                 titok_backbone='vit_base_patch16_224.mae',
                 titok_num_latent_tokens=32,
                 titok_patch_size=256):
        super().__init__()

        # High-res branch: standard Swin OR TiTok-based
        if use_titok:
            self.high_branch = HighResBranchTiTok(
                in_ch=high_in_ch,
                out_dim=high_out,
                patch_size=titok_patch_size,
                num_latent_tokens=titok_num_latent_tokens,
                pretrained=True,
                backbone=titok_backbone
            )
        else:
            self.high_branch = HighResBranch(
                out_dim=high_out,
                in_ch=high_in_ch,
                swin_model=swin_model
            )

        self.med_branch = MediumResBranchAttention(out_dim=med_out, in_ch=med_in_ch)
        self.static_branch = StaticBranch(out_dim=static_out, in_ch=static_in_ch)
        self.zone_branch = ZoneEmbeddingBlock(num_zones, static_out)

        # Output configuration
        self.output_size = output_size
        self.output_channels = output_channels
        self.decoder_channels = decoder_channels

        total_in = high_out + med_out + static_out + static_out
        H, W = output_size
        proj_pixels = decoder_channels * H * W
        self.fc = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_pixels),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(decoder_channels, output_channels, kernel_size=1)
        )

    def forward(self, x_high, x_med, x_static, x_cond):
        f_high = self.high_branch(x_high)
        f_med = self.med_branch(x_med)
        f_static = self.static_branch(x_static)
        f_embed = self.zone_branch(x_cond.long())

        fusion = torch.cat([f_high, f_med, f_static, f_embed], dim=1)

        out = self.fc(fusion)
        B = out.shape[0]
        H, W = self.output_size
        out = out.view(B, self.decoder_channels, H, W)
        out = self.decoder(out)
        return out