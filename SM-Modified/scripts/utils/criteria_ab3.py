"""
Multi-Stage Loss Function - Ab3 Ablation
=========================================
Modified loss for ablation testing:

Change: MSE(Magnitude) -> L1(Magnitude^0.3)

Rationale:
- L1 is less sensitive to outliers than MSE
- Power compression (mag^0.3) gives more weight to small magnitudes
- Small magnitudes are perceptually important (quiet sounds, consonants)
- This combination is closer to what CMGAN and MP-SENet use

Loss Formula:
SI-SDR + 10.0 * L1(R,I) + 3.0 * L1(Magnitude^0.3)
"""

import torch
import torch.nn.functional as F
from utils.pipeline_modules import Resynthesizer
from configs import MODEL_CONFIG


def si_sdr(estimated, target, eps=1e-8):
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    if estimated.dim() == 3:
        estimated = estimated.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)

    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)

    alpha = (torch.sum(estimated * target, dim=1, keepdim=True) /
            (torch.sum(target ** 2, dim=1, keepdim=True) + eps))
    target_scaled = alpha * target
    noise = estimated - target_scaled

    si_sdr_val = (torch.sum(target_scaled ** 2, dim=1) /
                  (torch.sum(noise ** 2, dim=1) + eps))
    return 10 * torch.log10(si_sdr_val + eps).mean()


def si_sdr_loss(estimated, target):
    """SI-SDR loss (negative for minimization)."""
    return -si_sdr(estimated, target)


def l1_loss_complex(est, ref):
    """L1 loss on real and imaginary components."""
    return F.l1_loss(est[:, 0], ref[:, 0]) + F.l1_loss(est[:, 1], ref[:, 1])


def magnitude_l1_loss(est, ref, eps=1e-8, power=0.3):
    """
    L1 loss on COMPRESSED magnitude spectrum.

    Args:
        est: Estimated complex spectrum [B, 2, F, T]
        ref: Reference complex spectrum [B, 2, F, T]
        eps: Small value for numerical stability
        power: Compression power (default 0.3, same as CMGAN)

    Key differences from MSE version:
    1. L1 instead of MSE: More robust to outliers
    2. Always compressed: mag^0.3 reduces dynamic range

    Why this helps:
    - Raw magnitude range: 0.001 to 100 (huge dynamic range)
    - MSE squares errors: (100-99)^2 = 1, (0.01-0.001)^2 = 0.00008
    - This makes MSE ignore small magnitude errors
    - Compression + L1: |4.0-3.98| = 0.02, |0.06-0.01| = 0.05
    - Small magnitudes now contribute meaningfully to loss
    """
    est_mag = torch.sqrt(est[:, 0]**2 + est[:, 1]**2 + eps)
    ref_mag = torch.sqrt(ref[:, 0]**2 + ref[:, 1]**2 + eps)

    # Always apply power compression
    est_mag_compressed = est_mag ** power
    ref_mag_compressed = ref_mag ** power

    return F.l1_loss(est_mag_compressed, ref_mag_compressed)


class LossFunction(object):
    """
    Multi-Stage Loss Function - Ab3 Version

    SI-SDR (time-domain) + L1 (complex) + L1 (compressed magnitude)

    Formula: SI-SDR + 10.0 * L1(R,I) + 3.0 * L1(Mag^0.3)

    Changes from original:
    - MSE(Magnitude) -> L1(Magnitude^0.3)
    - Compression is always on (not configurable)
    """

    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)

        # Fixed compression power (like CMGAN)
        self.mag_compression_power = MODEL_CONFIG.get('mag_compression_power', 0.3)

        print(f"[LossFunction-Ab3] Using L1 + Compressed magnitude loss (power={self.mag_compression_power})")

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples):
        # Apply mask
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask

        # Time-domain loss (SI-SDR)
        est_wave = self.resynthesizer(est_masked, mix)
        lbl_wave = self.resynthesizer(lbl_masked, mix)
        T = n_samples[0].item()
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        loss_sisdr = si_sdr_loss(est_wave, lbl_wave)

        # Complex component loss (L1 on real/imaginary)
        loss_mae = l1_loss_complex(est_masked, lbl_masked)

        # Magnitude loss: L1 on compressed magnitude (key change!)
        loss_mag = magnitude_l1_loss(
            est_masked, lbl_masked,
            power=self.mag_compression_power
        )

        # Combined loss: SI-SDR + 10.0 * L1(R,I) + 3.0 * L1(Mag^0.3)
        return loss_sisdr + 10.0 * loss_mae + 3.0 * loss_mag
