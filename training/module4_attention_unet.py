"""
MODULE 4 — Attention U-Net
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
~31-35M parameter deterministic regression model.
Input:  (B, 9, 256, 256)  — VV, VH, 6 reference optical bands, ref cloud mask
Output: (B, 6, 256, 256)  — reconstructed optical bands [0,1]

----------------------------------------------------------------
WHY DETERMINISTIC (NOT GAN, NOT DIFFUSION) AS THE PRIMARY MODEL
----------------------------------------------------------------
This is a literature-checked, defensible choice, not a default:

- GAN/diffusion methods for SAR-to-optical synthesis are optimized and
  benchmarked on PERCEPTUAL quality (FID/LPIPS/SSIM-on-image) — they
  produce visually plausible, sharp-looking output, but are not
  optimized for per-pixel numerical correctness.
- Documented failure modes of GANs for this exact task include training
  instability, mode collapse, and color/spectral distortion — i.e.
  visually convincing but spectrally wrong, which would directly corrupt
  any NDWI/MNDWI/NDVI value computed from the output.
- Diffusion models, despite beating GANs on perceptual benchmarks, are
  generative samplers: the same input can yield slightly different
  output pixel values across runs, since they sample from a learned
  distribution rather than deterministically regressing a value. For
  DSAS shoreline transects or flood-extent measurement, that sampling
  variance would be indistinguishable from real coastal change — an
  unacceptable source of noise for a downstream measurement pipeline.
- Our downstream task is QUANTITATIVE spectral-index measurement
  (erosion rate, flood extent in physical units), not visualization.
  A deterministic regression model trained with reconstruction loss
  directly optimizes for the metric that matters: pixel-level spectral
  accuracy, at the cost of potentially blurrier textures than a GAN.
- The GAN is still implemented (module5_gan_baseline.py) and trained on
  IDENTICAL data/splits as an honest ablation — "deterministic vs.
  adversarial, evaluated on index fidelity" is itself a paper
  contribution, regardless of which one wins.

Architecture unchanged from the prior verified version other than
input size confirmation at 256x256 (locked decision).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Standard additive attention gate (Oktay et al., Attention U-Net)."""

    def __init__(self, in_ch_gate, in_ch_skip, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(in_ch_gate, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(in_ch_skip, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[-2:] != s.shape[-2:]:
            g = nn.functional.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.psi(self.relu(g + s))
        return skip * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=6, base_ch=64):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.att4 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.att3 = AttentionGate(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.att2 = AttentionGate(base_ch * 2, base_ch * 2, base_ch)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.att1 = AttentionGate(base_ch, base_ch, base_ch // 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)
        self.out_activation = nn.Sigmoid()  # output in [0,1], matches normalized reflectance target

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        a4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, a4], dim=1))

        d3 = self.up3(d4)
        a3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, a3], dim=1))

        d2 = self.up2(d3)
        a2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, a2], dim=1))

        d1 = self.up1(d2)
        a1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, a1], dim=1))

        out = self.out_conv(d1)
        return self.out_activation(out)


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE 4 — Attention U-Net Verification")
    print("=" * 50)

    model = AttentionUNet(in_channels=9, out_channels=6, base_ch=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
    print("NOTE: if your paper states a specific target (e.g. ~35M), tune "
          "base_ch and re-run this file until it matches, then lock that "
          "value before training — the reported parameter count must be "
          "exact, not approximate.")

    dummy = torch.randn(2, 9, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expect (2, 6, 256, 256))")
    assert out.shape == (2, 6, 256, 256), "Output shape mismatch!"
    assert not torch.isnan(out).any(), "NaN in output!"
    print("Forward pass verified ✓")
