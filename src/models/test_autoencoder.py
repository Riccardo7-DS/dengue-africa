"""
Quick smoke test for the VAE autoencoder (no real data required).
Tests forward pass, backward pass, and shape correctness.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from models.training_autoencoder import (
    VAEWithTrainableResize,
    InputUpsampler,
    OutputDownsampler,
    masked_mse_loss,
    normalize_to_neg_one_one,
    denormalize_from_neg_one_one,
    ensure_bchw_1x86x86,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── helpers ────────────────────────────────────────────────────────────────────

def make_batch(B=2, H=86, W=86, nan_frac=0.1):
    """Synthetic batch shaped [B, 1, H, W] with some NaNs."""
    x = torch.randn(B, 1, H, W)
    mask = torch.rand_like(x) < nan_frac
    x[mask] = float("nan")
    return x.to(DEVICE)


def build_vae(latent_channels=4, block_out_channels=(64, 128, 256), layers_per_block=2):
    vae = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        latent_channels=latent_channels,
        down_block_types=("DownEncoderBlock2D",) * 3,
        up_block_types=("UpDecoderBlock2D",) * 3,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=32,
    )
    return VAEWithTrainableResize(vae).to(DEVICE)


# ── tests ──────────────────────────────────────────────────────────────────────

def test_submodule_shapes():
    print("\n[1] Upsampler / Downsampler shapes ...")
    x = torch.randn(2, 1, 86, 86).to(DEVICE)
    up = InputUpsampler().to(DEVICE)
    out_up = up(x)
    assert out_up.shape == (2, 1, 128, 128), f"Unexpected upsampler output: {out_up.shape}"

    down = OutputDownsampler().to(DEVICE)
    x128 = torch.randn(2, 1, 128, 128).to(DEVICE)
    out_down = down(x128)
    assert out_down.shape == (2, 1, 86, 86), f"Unexpected downsampler output: {out_down.shape}"
    print("   PASSED")


def test_forward_pass():
    print("\n[2] VAEWithTrainableResize forward pass ...")
    model = build_vae()
    model.eval()
    B = 2
    x = torch.randn(B, 1, 86, 86).to(DEVICE)

    with torch.no_grad():
        recon, post_mean, post_logvar = model(x)

    assert recon.shape == (B, 1, 86, 86), f"recon shape wrong: {recon.shape}"
    assert post_mean.shape[0] == B, f"posterior mean batch dim wrong: {post_mean.shape}"
    assert post_logvar.shape == post_mean.shape, "mean / logvar shape mismatch"
    print(f"   recon: {recon.shape}  mean: {post_mean.shape}  logvar: {post_logvar.shape}")
    print("   PASSED")


def test_backward_pass():
    print("\n[3] Backward pass (gradient flow) ...")
    model = build_vae()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = make_batch(B=2)
    valid_mask = (~torch.isnan(x)).bool()
    x_clean = torch.nan_to_num(x, nan=-2.0)
    x_norm = normalize_to_neg_one_one(x_clean, data_min=0.0, data_max=100.0)

    recon, post_mean, post_logvar = model(x_norm)
    recon_loss = masked_mse_loss(recon, x_norm, valid_mask)
    kl_loss = -0.5 * torch.mean(1 + post_logvar - post_mean.pow(2) - post_logvar.exp())
    loss = recon_loss + 0.01 * kl_loss

    assert not torch.isnan(loss), f"Loss is NaN!"
    loss.backward()
    optimizer.step()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed!"
    print(f"   loss={loss.item():.6f}  recon={recon_loss.item():.6f}  kl={kl_loss.item():.6f}")
    print("   PASSED")


def test_masked_mse():
    print("\n[4] masked_mse_loss ...")
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(DEVICE)
    target = torch.tensor([[1.0, float("nan")], [3.0, 4.0]]).to(DEVICE)
    valid = ~torch.isnan(target)
    loss = masked_mse_loss(pred, target, valid)
    assert not torch.isnan(loss), "masked_mse_loss returned NaN"
    expected = ((1-1)**2 + (3-3)**2 + (4-4)**2) / 3
    assert abs(loss.item() - expected) < 1e-5
    print(f"   loss={loss.item():.6f}  (expected {expected:.6f})")
    print("   PASSED")


def test_normalize_roundtrip():
    print("\n[5] normalize / denormalize roundtrip ...")
    x = torch.linspace(0, 100, 50).to(DEVICE)
    xn = normalize_to_neg_one_one(x, 0.0, 100.0)
    xr = denormalize_from_neg_one_one(xn, 0.0, 100.0)
    assert torch.allclose(x, xr, atol=1e-4), "Roundtrip failed"
    print("   PASSED")


def test_ensure_bchw():
    print("\n[6] ensure_bchw_1x86x86 ...")
    shapes_ok = [
        (86, 86),
        (1, 86, 86),
        (4, 1, 86, 86),
    ]
    for s in shapes_ok:
        t = torch.zeros(*s)
        out = ensure_bchw_1x86x86(t)
        assert out.shape[1] == 1 and out.shape[2] == 86 and out.shape[3] == 86, \
            f"Bad output shape {out.shape} for input {s}"
    print("   PASSED")


def test_amp_forward():
    print("\n[7] AMP (mixed precision) forward pass ...")
    model = build_vae()
    model.eval()
    x = torch.randn(2, 1, 86, 86).to(DEVICE)
    with torch.amp.autocast(device_type=DEVICE.type):
        recon, post_mean, post_logvar = model(x)
    assert recon.shape == (2, 1, 86, 86)
    print("   PASSED")


# ── run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_submodule_shapes,
        test_forward_pass,
        test_backward_pass,
        test_masked_mse,
        test_normalize_roundtrip,
        test_ensure_bchw,
        test_amp_forward,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"   FAILED: {e}")
            failures.append((t.__name__, e))

    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED {len(failures)}/{len(tests)} tests:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")
