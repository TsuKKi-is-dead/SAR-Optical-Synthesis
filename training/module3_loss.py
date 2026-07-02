"""
MODULE 3 — Loss Function
SAR-Guided Optical Reconstruction Pipeline (merged)
============================================================
COMPOSITE LOSS: 0.6 * L1 + 0.3 * Perceptual(VGG16) + 0.1 * SSIM
Applied UNIFORMLY across the whole output by default. No region-weighting
split (see below for why).

----------------------------------------------------------------
NEW: OPTIONAL PER-BAND L1 WEIGHTING (experiment, matches module5)
----------------------------------------------------------------
module8_bsi_diagnostic.py found BSI's weak R^2 traces to B2 (Blue) and
B4 (Red) being poorly reconstructed relative to their low natural
variance, NOT to B11 (SWIR1) as originally assumed. This adds an
optional band_weights mechanism to the L1 term ONLY (perceptual and
SSIM remain unweighted/global — see rationale below), using the exact
same weighting convention as module5_gan_baseline.py so a band_weights
list means the same thing for both models.

Default band_weights=None reproduces the EXACT original unweighted
behavior used for all previously reported U-Net results — nothing
changes unless explicitly passed.

Why only the L1 term gets band weights, not perceptual/SSIM:
- Perceptual loss operates on VGG features computed from a 3-channel
  RGB proxy (B2,B3,B4) as a single fused tensor — there's no clean way
  to up-weight "B4's contribution" inside VGG's feature space, since
  the conv layers mix channels immediately. Band-weighting would
  require reweighting after the fact in a way that doesn't have a
  principled meaning.
- SSIM is computed per-channel internally but is about *structural*
  similarity (local contrast/luminance patterns), not direct pixel
  value accuracy — the BSI diagnostic finding was about absolute
  reconstruction error (RMSE) relative to variance, which is what L1
  directly targets. Band-weighting SSIM would conflate two different
  problems.
- This keeps the experiment narrow and interpretable: "we up-weighted
  L1 on the bands the diagnostic flagged" is a clean, defensible
  sentence for a methods section. Spreading the weighting across all
  three loss terms would make it harder to attribute any resulting
  change to a specific cause.

----------------------------------------------------------------
WHY NO REGION-WEIGHTING SPLIT (read this before changing it back)
----------------------------------------------------------------
An earlier version of this pipeline (designed for a different task —
same-day synthetic cloud-pasting) used an 0.8/0.2 masked-vs-clear region
split: 0.8 weight on synthetically-blacked-out pixels (the "hard"
reconstruction region), 0.2 weight on pixels that were never touched (the
"easy" region, kept mostly to prevent the model distorting known-good
pixels). That split exists ONLY because, in that task, part of the image
is genuinely already known and part is genuinely hidden — without
weighting, a model trained on mostly-clear patches just learns to copy
input to output and barely engages with the hard region.

In THIS pipeline's actual task (triplets), there is no such split to make:
the entire target is a real optical scene from a genuinely different date
that the model never saw as input. There is no "easy, already-known"
sub-region left to protect or de-weight — the WHOLE output is the
reconstruction task. Reintroducing an 0.8/0.2 split here would copy the
old structure without the problem it was designed to solve. If asked by
a reviewer: "loss is applied uniformly because our reconstruction target
is fully novel relative to the input reference, unlike inpainting
formulations where part of the target is already known."

The reference scene's own cloud mask (channel 8 of the input) is NOT used
as a loss weight. It's used only as an input feature, telling the model
where its optical reference is unreliable so it can lean on SAR more in
those regions — that's a legitimate, defensible use. Using it to weight
the LOSS instead would imply "the target will be harder to predict where
the reference was cloudy," which has no causal basis (the reference's
cloud pattern and the target's prediction difficulty are not linked) and
would not survive review scrutiny.

The NEW band_weights mechanism above is a different axis (per-channel,
not per-region) and does not reopen this question — it's about which
of the 6 fixed optical bands gets more L1 attention, not about which
spatial pixels do.

----------------------------------------------------------------
WHY THIS COMPOSITE (L1 + perceptual + SSIM), kept from the original design
----------------------------------------------------------------
L1 (0.6):         pixel-level accuracy — most directly tied to NDWI/NDVI/
                  MNDWI index accuracy, which is the actual downstream use.
Perceptual (0.3): VGG16 feature-matching — without it, L1-only regression
                  tends toward blurry, averaged outputs (a documented,
                  known failure mode of pixel losses alone). Forces
                  realistic coastal/water texture even though the U-Net
                  itself is trained deterministically (see module4 header
                  for the deterministic-vs-generative reasoning; this
                  loss is shared conceptually with the GAN's L1 term but
                  the U-Net has no adversarial term).
SSIM (0.1):       structural similarity — catches boundary-shift errors
                  (e.g., shoreline at the right average value but shifted
                  a few pixels), which L1 alone can miss. Directly
                  relevant to downstream shoreline-position accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

OPTICAL_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


def _make_band_weight_tensor(band_weights, device, dtype=torch.float32):
    """
    Identical convention to module5_gan_baseline.py's helper of the same
    name — kept as a free function (not a method) so both modules can
    import from a single source if desired, though it's duplicated here
    for now to avoid a cross-module import dependency for such a small
    helper. If you change the renormalization logic, change it in BOTH
    files or the U-Net and GAN band_weights will stop meaning the same
    thing.
    """
    if band_weights is None:
        return None
    if len(band_weights) != 6:
        raise ValueError(
            f"band_weights must have exactly 6 values (one per "
            f"{OPTICAL_BAND_ORDER}), got {len(band_weights)}"
        )
    w = torch.tensor(band_weights, dtype=dtype, device=device)
    w = w * (6.0 / w.sum())  # renormalize so mean weight = 1.0, keeps l1_weight=0.6 comparable
    return w.view(1, 6, 1, 1)


def weighted_l1(pred, target, band_weights_tensor):
    if band_weights_tensor is None:
        return F.l1_loss(pred, target)
    diff = (pred - target).abs()
    return (diff * band_weights_tensor).mean()


class PerceptualLoss(nn.Module):
    """VGG16 feature-matching loss. Frozen pretrained weights.
    Uses first 3 channels (B2,B3,B4) of the 6-channel optical output as
    an RGB proxy, since VGG16 expects 3-channel input."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize_for_vgg(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred_rgb = pred[:, :3, :, :]
        target_rgb = target[:, :3, :, :]
        pred_norm = self.normalize_for_vgg(pred_rgb)
        target_norm = self.normalize_for_vgg(target_rgb)
        pred_feat = self.features(pred_norm)
        target_feat = self.features(target_norm)
        return F.mse_loss(pred_feat, target_feat)


class SSIMLoss(nn.Module):
    """Returns 1 - SSIM (so it behaves as a loss to minimize).
    Window size 11, standard for satellite imagery."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.window = self._create_window(window_size)

    def _create_window(self, size):
        sigma = 1.5
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g)
        return window.unsqueeze(0).unsqueeze(0)

    def _ssim(self, x, y):
        B, C, H, W = x.shape
        window = self.window.to(x.device).expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=pad, groups=C)
        mu_y = F.conv2d(y, window, padding=pad, groups=C)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy

        numerator = (2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)
        denominator = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        return (numerator / denominator).mean()

    def forward(self, pred, target):
        return 1.0 - self._ssim(pred, target)


class CompositeLoss(nn.Module):
    """
    Main loss for the Attention U-Net (and conceptually mirrored by the
    GAN's L1 term, for a fair ablation).

    total = 0.6 * L1 + 0.3 * Perceptual + 0.1 * SSIM, uniform over the
    full (B, 6, 256, 256) output by default. No region weighting (see
    module docstring for why).

    band_weights: optional list of 6 floats, one per
    [B2, B3, B4, B8, B11, B12], applied ONLY to the L1 term (see module
    docstring for why perceptual/SSIM are left unweighted). None
    (default) = unweighted L1, identical to all previously reported
    U-Net results. Pass e.g. [2.0, 1.0, 2.0, 1.0, 1.0, 1.0] to up-weight
    B2/B4 as the BSI diagnostic experiment — same convention as
    module5_gan_baseline.py's band_weights argument.
    """

    def __init__(self, l1_weight=0.6, perceptual_weight=0.3, ssim_weight=0.1,
                 band_weights=None):
        super().__init__()
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.band_weights = band_weights
        self._bw_tensor = None  # lazily built on first forward(), once we know the device

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 6, H, W) model output [0,1]
            target: (B, 6, H, W) ground-truth optical [0,1]
        Returns:
            total_loss: scalar
            loss_dict: breakdown for logging
        """
        if self.band_weights is not None and self._bw_tensor is None:
            self._bw_tensor = _make_band_weight_tensor(self.band_weights, pred.device, pred.dtype)

        l1 = weighted_l1(pred, target, self._bw_tensor)
        perc = self.perceptual(pred, target)
        ssim_l = self.ssim(pred, target)

        total = (self.l1_weight * l1 +
                 self.perceptual_weight * perc +
                 self.ssim_weight * ssim_l)

        loss_dict = {
            "total": total.item(),
            "l1": l1.item(),
            "perceptual": perc.item(),
            "ssim_loss": ssim_l.item(),
        }
        return total, loss_dict


if __name__ == "__main__":
    print("=" * 50)
    print("MODULE 3 — Loss Function Verification")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B = 4
    pred = torch.rand(B, 6, 256, 256, requires_grad=True).to(device)
    target = torch.rand(B, 6, 256, 256).to(device)

    print("\n--- Unweighted (default, matches all previously reported results) ---")
    criterion = CompositeLoss().to(device)
    loss, loss_dict = criterion(pred, target)
    for k, v in loss_dict.items():
        print(f"  {k:12s}: {v:.6f}")
    loss.backward()
    assert pred.grad is not None, "No gradient computed!"
    print(f"  grad_norm   : {pred.grad.norm():.6f}")

    print("\n--- B2/B4 up-weighted (BSI diagnostic experiment) ---")
    pred2 = torch.rand(B, 6, 256, 256, requires_grad=True).to(device)
    criterion_w = CompositeLoss(band_weights=[2.0, 1.0, 2.0, 1.0, 1.0, 1.0]).to(device)
    loss_w, loss_dict_w = criterion_w(pred2, target)
    for k, v in loss_dict_w.items():
        print(f"  {k:12s}: {v:.6f}")
    loss_w.backward()
    assert pred2.grad is not None, "No gradient computed!"
    print(f"  grad_norm   : {pred2.grad.norm():.6f}")

    print("\nBackprop works for both configurations ✓")
    print("=== Module 3 Complete ===")