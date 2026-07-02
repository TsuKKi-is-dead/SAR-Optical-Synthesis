"""
MODULE 5 — GAN (now PRIMARY model — see module7 docstring / paper notes
for the empirical reasoning: GAN outperformed the deterministic U-Net
on index-level R^2 across NDVI/NDWI/MNDWI in the 100-epoch run, despite
U-Net's theoretical determinism advantage. U-Net is retained as the
ablation/baseline.)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Pix2pix-style conditional GAN, 9-channel input / 6-channel output.
Generator REUSES the AttentionUNet architecture from module4 — this
keeps the comparison fair: the only variable is "adversarial training
vs. pure deterministic regression," not also a different architecture.

----------------------------------------------------------------
NEW: OPTIONAL PER-BAND L1 WEIGHTING (experiment, matches module3)
----------------------------------------------------------------
module8_bsi_diagnostic.py found BSI's weak R^2 traces to B2 (Blue) and
B4 (Red) being poorly reconstructed relative to their low natural
variance, NOT to B11 (SWIR1) as originally assumed. This adds the same
optional band_weights mechanism used in module3_loss.py to the GAN's
L1 term, so the experiment can be tested on the actual primary model.

Default band_weights=None reproduces the EXACT original unweighted
behavior used for the reported Run 2 GAN results (R^2: NDVI 0.73, NDWI
0.78, MNDWI 0.80, BSI 0.13) — nothing changes unless explicitly passed.
"""

import torch
import torch.nn as nn

from module4_attention_unet import AttentionUNet

OPTICAL_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator, standard pix2pix design."""

    def __init__(self, in_channels=9 + 6):  # condition (9ch input) + image (6ch real or fake)
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, norm=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, condition, image):
        x = torch.cat([condition, image], dim=1)
        return self.model(x)


def _make_band_weight_tensor(band_weights, device):
    if band_weights is None:
        return None
    if len(band_weights) != 6:
        raise ValueError(
            f"band_weights must have exactly 6 values (one per "
            f"{OPTICAL_BAND_ORDER}), got {len(band_weights)}"
        )
    w = torch.tensor(band_weights, dtype=torch.float32, device=device)
    w = w * (6.0 / w.sum())  # renormalize so mean weight = 1.0 -> l1_weight=100 stays comparable
    return w.view(1, 6, 1, 1)


def weighted_l1(pred, target, band_weights_tensor):
    if band_weights_tensor is None:
        return nn.functional.l1_loss(pred, target)
    diff = (pred - target).abs()
    return (diff * band_weights_tensor).mean()


def gan_training_step(generator, discriminator, opt_g, opt_d, batch, l1_weight=100.0,
                       device="cuda", band_weights=None):
    """
    Single pix2pix training step.
    l1_weight=100 is the standard pix2pix default — forces the GAN to
    also respect pixel accuracy, not just adversarial realism.

    band_weights: optional list of 6 floats (see module docstring). None
    (default) = unweighted L1, identical to the originally reported
    Run 2 GAN results. Pass e.g. [2.0, 1.0, 2.0, 1.0, 1.0, 1.0] to
    up-weight B2/B4 as the BSI diagnostic experiment.
    """
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    bw_tensor = _make_band_weight_tensor(band_weights, device)

    # --- Train Discriminator ---
    opt_d.zero_grad()
    with torch.no_grad():
        fake = generator(inputs)
    pred_real = discriminator(inputs, targets)
    pred_fake = discriminator(inputs, fake.detach())
    loss_d_real = nn.functional.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    loss_d_fake = nn.functional.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    opt_d.step()

    # --- Train Generator ---
    opt_g.zero_grad()
    fake = generator(inputs)
    pred_fake_for_g = discriminator(inputs, fake)
    loss_g_adv = nn.functional.binary_cross_entropy_with_logits(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
    loss_g_l1 = weighted_l1(fake, targets, bw_tensor)
    loss_g = loss_g_adv + l1_weight * loss_g_l1
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_g_l1": loss_g_l1.item(),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE 5 — GAN Verification")
    print("=" * 50)

    device = torch.device("cpu")  # forced cpu for sandbox smoke test
    generator = AttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    discriminator = PatchDiscriminator(in_channels=9 + 6).to(device)

    n_gen = sum(p.numel() for p in generator.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator params:     {n_gen:,}")
    print(f"Discriminator params: {n_disc:,}")

    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    dummy_batch = {
        "input": torch.randn(2, 9, 256, 256),
        "target": torch.rand(2, 6, 256, 256),
    }

    print("\n--- Unweighted (default, matches original reported results) ---")
    stats = gan_training_step(generator, discriminator, opt_g, opt_d, dummy_batch, device=device)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- B2/B4 up-weighted (experiment) ---")
    stats_w = gan_training_step(generator, discriminator, opt_g, opt_d, dummy_batch, device=device,
                                  band_weights=[2.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    for k, v in stats_w.items():
        print(f"  {k}: {v:.4f}")

    print("\nGAN module verified ✓")