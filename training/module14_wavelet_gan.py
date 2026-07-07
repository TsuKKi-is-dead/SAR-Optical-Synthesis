"""
MODULE 14 — Wavelet-Domain GAN (WFLM-GAN-inspired baseline)
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Reimplements the core mechanism of Li et al. [WFLM-GAN, TGRS 2022] on
top of our EXISTING Attention U-Net / PatchGAN pipeline, rather than
building their full two-stage architecture from scratch. Per their own
ablation (Table III), the wavelet-feature generator (WG) alone is the
dominant contributor (SSIM 65.0% -> 82.2%, PSNR 20.9 -> 24.5 dB), far
larger than the wavelet discriminator (WD) or coloring network (CG)
individually. We reimplement WG + their EDGE loss (Eq. 6), and
explicitly do NOT reimplement WD or CG — see paper Methods section
wording note at the bottom of this file for how to describe this
honestly.

----------------------------------------------------------------
WHAT CHANGES vs. our existing GAN (module5_gan_baseline.py)
----------------------------------------------------------------
- Generator: SAME AttentionUNet skeleton (encoder/decoder/attention
  gates reused verbatim from module4), but the output head is changed
  from "6 channels + Sigmoid" to "24 channels (4 wavelet subbands x 6
  optical bands) at HALF spatial resolution (128x128), with a SPLIT
  activation": Sigmoid on the LL block (matches [0,1] pixel-average
  range), scaled Tanh on LH/HL/HH (matches [-0.5,0.5] range) — see
  module13's verified value ranges.
- The 24-channel wavelet prediction is passed through haar_iwt (module13)
  to reconstruct the final 6-channel, 256x256 optical image. This IS
  the generator's forward() output — everything downstream (loss,
  discriminator, evaluation) sees a normal 6-channel image, unchanged.
- Discriminator: UNCHANGED, reuses PatchDiscriminator from module5
  verbatim (their WD contributed far less than WG in the ablation, and
  a faithful wavelet-feature discriminator was judged infeasible on
  this timeline).
- Loss: existing L1 + adversarial (module5's gan_training_step logic)
  PLUS the EDGE loss term (module13.edge_loss) with weight
  lambda_edge, applied on the reconstructed 6-channel image (not the
  wavelet coefficients) so it's directly comparable across model
  variants using the same evaluation suite (module6).

----------------------------------------------------------------
WHY HALF-RESOLUTION INTERNAL PREDICTION
----------------------------------------------------------------
Haar DWT halves spatial dimensions. The AttentionUNet's decoder
already upsamples back to 256x256 by design (symmetric with its
encoder's 4 pooling stages); to get a 128x128 wavelet-coefficient map
instead, we simply drop the LAST upsampling stage's output resolution
by running the same decoder ladder one stage short and taking the
128x128 feature map directly as the wavelet head's input. This keeps
~90% of the original AttentionUNet code path identical (same encoder,
same attention gates, same first three decoder stages) and only
replaces the FINAL decoder stage + output conv, which is the smallest,
cheapest part to reimplement correctly in the available time.

Usage (drop-in replacement for train_gan() in module7_train.py):
    from module14_wavelet_gan import (
        WaveletAttentionUNet, wavelet_gan_training_step,
    )
    generator = WaveletAttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    # then reuse module5.PatchDiscriminator + the same train_gan() loop
    # structure in module7, calling wavelet_gan_training_step instead of
    # gan_training_step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from module4_attention_unet import ConvBlock, AttentionGate
from module5_gan_baseline import PatchDiscriminator  # reused verbatim, unchanged
from module13_wavelet_utils import haar_iwt, EdgeExtractor, edge_loss

OPTICAL_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


class WaveletAttentionUNet(nn.Module):
    """
    Same encoder + attention-gated decoder ladder as module4.AttentionUNet
    for the first 3 decoder stages (256->128->64->32->16 bottleneck, back
    up to 32x32). The FINAL decoder stage (32x32 -> 64x64 -> ... -> 256x256
    in the original) is replaced: instead of going all the way back to
    256x256 pixel space, we stop at 128x128 and predict 4*out_channels
    wavelet-subband channels there directly, then reconstruct to
    256x256 pixel space via inverse Haar wavelet transform (module13).

    forward() returns the FINAL 256x256, out_channels-channel image
    (post-IWT) — identical output contract to AttentionUNet, so it's a
    drop-in replacement everywhere downstream (loss, discriminator,
    evaluation, module10/11/12 inference scripts).
    """

    def __init__(self, in_channels=9, out_channels=6, base_ch=64):
        super().__init__()
        self.out_channels = out_channels

        # --- encoder: identical to module4.AttentionUNet ---
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        # --- decoder stages 4,3,2: identical to module4.AttentionUNet ---
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.att4 = AttentionGate(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.att3 = AttentionGate(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.att2 = AttentionGate(base_ch * 2, base_ch * 2, base_ch)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        # NOTE: dec2 output is at 128x128 spatial resolution (H/2, W/2 of
        # the original 256x256 input) — this is exactly the resolution
        # Haar DWT/IWT operates at, so we stop the decoder ladder HERE
        # instead of running the original up1/att1/dec1 stage.

        # --- wavelet-coefficient head, replaces module4's up1/att1/dec1/out_conv ---
        wavelet_ch = out_channels * 4  # LL,LH,HL,HH stacked
        self.wavelet_head = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, wavelet_ch, 1),
        )

        self.iwt = None  # constructed lazily, needs out_channels (already known) — see below
        self._out_channels_for_iwt = out_channels

    def _split_activation(self, raw):
        """raw: (B, 4*C, H/2, W/2), channel order [LL(C),LH(C),HL(C),HH(C)]
        (must match module13.haar_iwt's expected ordering exactly)."""
        C = self.out_channels
        LL_raw = raw[:, 0 * C:1 * C]
        LH_raw = raw[:, 1 * C:2 * C]
        HL_raw = raw[:, 2 * C:3 * C]
        HH_raw = raw[:, 3 * C:4 * C]

        LL = torch.sigmoid(LL_raw)                      # matches [0,1] range
        LH = 0.5 * torch.tanh(LH_raw)                    # matches [-0.5,0.5] range
        HL = 0.5 * torch.tanh(HL_raw)
        HH = 0.5 * torch.tanh(HH_raw)

        return torch.cat([LL, LH, HL, HH], dim=1)

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
        d2 = self.dec2(torch.cat([d2, a2], dim=1))   # (B, base_ch*2, 128, 128)

        wavelet_raw = self.wavelet_head(d2)            # (B, 4*out_channels, 128, 128)
        wavelet_coeffs = self._split_activation(wavelet_raw)

        out = haar_iwt(wavelet_coeffs, channels=self.out_channels)  # (B, out_channels, 256, 256)
        return out


def _make_band_weight_tensor(band_weights, device):
    if band_weights is None:
        return None
    if len(band_weights) != 6:
        raise ValueError(
            f"band_weights must have exactly 6 values (one per "
            f"{OPTICAL_BAND_ORDER}), got {len(band_weights)}"
        )
    w = torch.tensor(band_weights, dtype=torch.float32, device=device)
    w = w * (6.0 / w.sum())
    return w.view(1, 6, 1, 1)


def weighted_l1(pred, target, band_weights_tensor):
    if band_weights_tensor is None:
        return F.l1_loss(pred, target)
    diff = (pred - target).abs()
    return (diff * band_weights_tensor).mean()


def wavelet_gan_training_step(generator, discriminator, edge_extractor,
                               opt_g, opt_d, batch, l1_weight=100.0,
                               edge_weight=10.0, device="cuda", band_weights=None):
    """
    Same overall structure as module5.gan_training_step, with ONE
    addition: an EDGE loss term (module13) added to the generator loss.

    edge_weight=10.0 is a starting point, not independently tuned on
    this dataset (no time budget for a proper sweep — see Li et al.
    Table IV for their own tuning context, which used far more data).
    If total loss magnitudes look unbalanced in early training logs
    (edge loss dominating or negligible vs. l1/adversarial), adjust
    this first before anything else.
    """
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    bw_tensor = _make_band_weight_tensor(band_weights, device)

    # --- Train Discriminator (unchanged from module5) ---
    opt_d.zero_grad()
    with torch.no_grad():
        fake = generator(inputs)
    pred_real = discriminator(inputs, targets)
    pred_fake = discriminator(inputs, fake.detach())
    loss_d_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
    loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
    loss_d = (loss_d_real + loss_d_fake) * 0.5
    loss_d.backward()
    opt_d.step()

    # --- Train Generator ---
    opt_g.zero_grad()
    fake = generator(inputs)
    pred_fake_for_g = discriminator(inputs, fake)
    loss_g_adv = F.binary_cross_entropy_with_logits(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
    loss_g_l1 = weighted_l1(fake, targets, bw_tensor)
    loss_g_edge = edge_loss(fake, targets, edge_extractor)

    loss_g = loss_g_adv + l1_weight * loss_g_l1 + edge_weight * loss_g_edge
    loss_g.backward()
    opt_g.step()

    return {
        "loss_d": loss_d.item(),
        "loss_g_adv": loss_g_adv.item(),
        "loss_g_l1": loss_g_l1.item(),
        "loss_g_edge": loss_g_edge.item(),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 14 — Wavelet-Domain GAN Verification")
    print("=" * 60)

    device = torch.device("cpu")  # forced cpu for sandbox smoke test

    generator = WaveletAttentionUNet(in_channels=9, out_channels=6, base_ch=64).to(device)
    discriminator = PatchDiscriminator(in_channels=9 + 6).to(device)
    edge_extractor = EdgeExtractor(channels=6).to(device)

    n_gen = sum(p.numel() for p in generator.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters())
    print(f"WaveletAttentionUNet generator params: {n_gen:,} (~{n_gen/1e6:.1f}M)")
    print(f"PatchDiscriminator params:             {n_disc:,}")
    print("(compare to plain AttentionUNet ~31-35M -- wavelet head is small, "
          "should be very close since encoder/most of decoder is identical)")

    dummy = torch.randn(2, 9, 256, 256)
    with torch.no_grad():
        out = generator(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}  (expect (2, 6, 256, 256))")
    assert out.shape == (2, 6, 256, 256), "Output shape mismatch!"
    assert not torch.isnan(out).any(), "NaN in output!"
    print("Forward pass shape/NaN check \u2713")

    # Full training step smoke test
    opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    dummy_batch = {
        "input": torch.randn(2, 9, 256, 256),
        "target": torch.rand(2, 6, 256, 256),
    }

    print("\n--- Training step smoke test (unweighted) ---")
    stats = wavelet_gan_training_step(
        generator, discriminator, edge_extractor, opt_g, opt_d, dummy_batch, device=device
    )
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Training step smoke test (B2/B4 up-weighted, matches module5/7 convention) ---")
    stats_w = wavelet_gan_training_step(
        generator, discriminator, edge_extractor, opt_g, opt_d, dummy_batch, device=device,
        band_weights=[2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
    )
    for k, v in stats_w.items():
        print(f"  {k}: {v:.4f}")

    print("\nWavelet GAN module verified \u2713")
    print("\n" + "=" * 60)
    print("PAPER METHODS-SECTION WORDING (use as a starting point):")
    print("=" * 60)
    print("""
  "As a stronger published baseline, we reimplement the core
  wavelet-feature-learning mechanism of [Li et al., WFLM-GAN, TGRS
  2022], which their own ablation study identifies as the dominant
  contributor to their method's performance (SSIM 65.0% -> 82.2% from
  this component alone). Our generator predicts Haar wavelet subband
  coefficients (LL/LH/HL/HH) rather than raw pixel values, which are
  then reconstructed via inverse wavelet transform; we additionally
  adopt their EDGE loss term. We do not reimplement their dual
  wavelet/multiscale-coloring discriminator or two-branch coloring
  network, as these contribute less individually per their own Table
  III and were not feasible to reproduce faithfully within our
  single-domain, 311-patch training regime; our existing 70x70
  PatchGAN discriminator (identical across all model variants
  reported) is retained for a controlled comparison."
""")
