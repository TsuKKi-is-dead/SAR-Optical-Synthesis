"""
MODULE 13 — Haar Wavelet Transform Utilities + EDGE Loss
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
Reimplements the two components of Li et al. [WFLM-GAN, TGRS 2022]
identified (via their own Table III ablation) as the highest-leverage
contributors to their result: the wavelet-feature-learning generator
(WG: their single biggest jump, SSIM 65.0% -> 82.2%) and the EDGE loss
(their Eq. 6). Does NOT reimplement their dual wavelet/color
discriminator split (WD) or multiscale coloring network (CG), which
their own ablation shows contribute far less individually and are not
feasible to build faithfully on our 311-patch, few-days timeline.

----------------------------------------------------------------
HAAR DWT/IWT — exact, invertible, average/difference convention
----------------------------------------------------------------
For each non-overlapping 2x2 pixel block (a=top-left, b=top-right,
c=bottom-left, d=bottom-right), all in the SAME channel:

    LL = (a + b + c + d) / 4        <- local average (low-frequency)
    LH = (a + b - c - d) / 4        <- vertical gradient
    HL = (a - b + c - d) / 4        <- horizontal gradient
    HH = (a - b - c + d) / 4        <- diagonal gradient

This is NOT the orthonormal Haar basis (that uses a 0.5 factor and
requires a matching transpose-conv inverse) — it is the simpler
average/difference convention, chosen because:
  1. LL lands in the exact same [0,1] range as the input pixels
     (average of 4 values in [0,1] is in [0,1]), so the generator can
     use a plain Sigmoid on the LL channels.
  2. LH/HL/HH land in [-0.5, 0.5], so a scaled Tanh works cleanly.
  3. The forward/inverse pair is verified exactly invertible below
     (round-trip test in __main__) — reconstruction is EXACT to
     float32 precision, not approximate.

Input/output channel convention: for C input channels, DWT output is
4*C channels ordered [LL(C channels), LH(C), HL(C), HH(C)] — i.e. all
LL bands first, then all LH bands, etc. (matches how the generator
head is split in module14).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt(x):
    """
    x: (B, C, H, W), H and W must be even.
    Returns: (B, 4*C, H//2, W//2), channel order [LL(C), LH(C), HL(C), HH(C)]
    """
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError(f"haar_dwt requires even H,W, got {H}x{W}")

    x = x.view(B, C, H // 2, 2, W // 2, 2)
    a = x[:, :, :, 0, :, 0]
    b = x[:, :, :, 0, :, 1]
    c = x[:, :, :, 1, :, 0]
    d = x[:, :, :, 1, :, 1]

    LL = (a + b + c + d) / 4.0
    LH = (a + b - c - d) / 4.0
    HL = (a - b + c - d) / 4.0
    HH = (a - b - c + d) / 4.0

    return torch.cat([LL, LH, HL, HH], dim=1)


def haar_iwt(coeffs, channels):
    """
    Inverse of haar_dwt. Exact reconstruction.
    coeffs: (B, 4*channels, H//2, W//2)
    Returns: (B, channels, H, W)
    """
    B, C4, H2, W2 = coeffs.shape
    C = channels
    if C4 != 4 * C:
        raise ValueError(f"Expected {4*C} channels (4*{C}), got {C4}")

    LL = coeffs[:, 0 * C:1 * C]
    LH = coeffs[:, 1 * C:2 * C]
    HL = coeffs[:, 2 * C:3 * C]
    HH = coeffs[:, 3 * C:4 * C]

    a = LL + LH + HL + HH
    b = LL + LH - HL - HH
    c = LL - LH + HL - HH
    d = LL - LH - HL + HH

    H, W = H2 * 2, W2 * 2
    out = torch.zeros(B, C, H, W, device=coeffs.device, dtype=coeffs.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


class EdgeExtractor(nn.Module):
    """
    Edge extraction following Li et al. Eq. 6: downsample the image,
    upsample it back, then subtract from the original. The residual is
    the "edge" (high-frequency detail lost by down/up-sampling).
    Applied per-channel (depthwise), same operation for both the
    generated and target images, then EDGE loss = L1(edge_pred, edge_gt).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(down, scale_factor=2, mode="bilinear", align_corners=False)
        if up.shape[-2:] != x.shape[-2:]:
            up = F.interpolate(up, size=x.shape[-2:], mode="bilinear", align_corners=False)
        edge = x - up
        return edge


def edge_loss(pred, target, extractor):
    """L1 loss between edge maps of pred and target. Both must be the
    SAME extractor instance (stateless here, but kept as a module for
    API consistency / future extension e.g. learnable edge kernels)."""
    edge_pred = extractor(pred)
    edge_target = extractor(target)
    return F.l1_loss(edge_pred, edge_target)


if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 13 — Haar DWT/IWT + EDGE loss verification")
    print("=" * 60)

    torch.manual_seed(0)
    B, C, H, W = 2, 6, 256, 256
    x = torch.rand(B, C, H, W)

    coeffs = haar_dwt(x)
    print(f"Input:  {x.shape}")
    print(f"DWT out: {coeffs.shape}  (expect ({B}, {4*C}, {H//2}, {W//2}))")
    assert coeffs.shape == (B, 4 * C, H // 2, W // 2)

    x_rec = haar_iwt(coeffs, channels=C)
    print(f"IWT out: {x_rec.shape}  (expect {x.shape})")
    assert x_rec.shape == x.shape

    max_err = (x - x_rec).abs().max().item()
    print(f"\nMax reconstruction error (should be ~0, float32 precision): {max_err:.2e}")
    assert max_err < 1e-5, "DWT/IWT round-trip is NOT exact — bug in transform!"
    print("Round-trip reconstruction EXACT \u2713")

    # sanity check on value ranges
    C_ = C
    LL = coeffs[:, 0:C_]
    LH = coeffs[:, C_:2*C_]
    HL = coeffs[:, 2*C_:3*C_]
    HH = coeffs[:, 3*C_:4*C_]
    print(f"\nLL range: [{LL.min():.3f}, {LL.max():.3f}]  (expect within [0,1])")
    print(f"LH range: [{LH.min():.3f}, {LH.max():.3f}]  (expect within [-0.5,0.5])")
    print(f"HL range: [{HL.min():.3f}, {HL.max():.3f}]  (expect within [-0.5,0.5])")
    print(f"HH range: [{HH.min():.3f}, {HH.max():.3f}]  (expect within [-0.5,0.5])")

    # gradient flow check
    x_req = torch.rand(B, C, H, W, requires_grad=True)
    coeffs2 = haar_dwt(x_req)
    rec2 = haar_iwt(coeffs2, channels=C)
    loss = rec2.sum()
    loss.backward()
    assert x_req.grad is not None and not torch.isnan(x_req.grad).any()
    print("\nGradient flows through DWT -> IWT correctly \u2713")

    # EDGE loss check
    extractor = EdgeExtractor(channels=C)
    pred = torch.rand(B, C, H, W, requires_grad=True)
    target = torch.rand(B, C, H, W)
    e_loss = edge_loss(pred, target, extractor)
    e_loss.backward()
    print(f"\nEDGE loss value (random inputs, sanity only): {e_loss.item():.4f}")
    assert pred.grad is not None
    print("EDGE loss backward works \u2713")

    print("\n=== Module 13 verification complete ===")
