"""
MODULE 5 — GAN Baseline (Ablation)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Pix2pix-style conditional GAN, SAME 9-channel input / 6-channel output
as the Attention U-Net. The generator REUSES the AttentionUNet
architecture from module4 — this keeps the ablation fair: the only
variable being tested is "adversarial training vs. pure deterministic
regression," not also a different architecture. This is what lets you
write "Table X: U-Net vs. GAN, identical architecture, identical data,
identical splits" in the paper.

This is NOT the primary model — see module4's docstring for why
deterministic regression was chosen as primary. This exists specifically
so you can show reviewers you tested the adversarial alternative rather
than dismissing it on literature alone.
"""

import torch
import torch.nn as nn

from module4_attention_unet import AttentionUNet


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


def gan_training_step(generator, discriminator, opt_g, opt_d, batch, l1_weight=100.0, device="cuda"):
    """
    Single pix2pix training step.
    l1_weight=100 is the standard pix2pix default — forces the GAN to
    also respect pixel accuracy, not just adversarial realism. Worth
    reporting sensitivity to this value, since it directly controls the
    spectral-fidelity-vs-realism tradeoff that the ablation exists to
    measure.
    """
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)

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
    loss_g_l1 = nn.functional.l1_loss(fake, targets)
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
    print("MODULE 5 — GAN Baseline Verification")
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

    stats = gan_training_step(generator, discriminator, opt_g, opt_d, dummy_batch, device=device)
    print("\nOne training step completed:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print("\nGAN baseline step verified ✓")
