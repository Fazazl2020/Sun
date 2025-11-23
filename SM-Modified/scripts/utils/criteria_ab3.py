"""
Multi-Stage Loss Function - Ab3 Ablation
=========================================
Evidence-based loss matching CMGAN (PESQ 3.41):

Loss = 0.9 * MSE(Mag) + 0.1 * MSE(R,I) + 0.2 * L1(Time)

Key differences from original:
1. Weight ratio FLIPPED: Magnitude dominant (0.9) vs Complex (0.1)
2. SI-SDR replaced with simple L1 time-domain loss
3. All spectral losses use MSE (same as CMGAN)

Original loss (wrong ratio):
  SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Mag)  → RI dominant (3:1)

New loss (CMGAN-style):
  0.9 * MSE(Mag) + 0.1 * MSE(R,I) + 0.2 * L1(Time)  → Mag dominant (9:1)
"""

import torch
import torch.nn.functional as F
from utils.pipeline_modules import Resynthesizer


def mse_loss_complex(est, ref):
    """MSE loss on real and imaginary components."""
    return F.mse_loss(est[:, 0], ref[:, 0]) + F.mse_loss(est[:, 1], ref[:, 1])


def magnitude_mse_loss(est, ref, eps=1e-8):
    """
    MSE loss on magnitude spectrum.

    Same as CMGAN - no compression, just raw MSE on magnitude.
    """
    est_mag = torch.sqrt(est[:, 0]**2 + est[:, 1]**2 + eps)
    ref_mag = torch.sqrt(ref[:, 0]**2 + ref[:, 1]**2 + eps)
    return F.mse_loss(est_mag, ref_mag)


def time_domain_l1_loss(est_wave, ref_wave):
    """
    L1 loss on time-domain waveform.

    CMGAN/MP-SENet use this instead of SI-SDR.
    Simple and direct - no scale invariance complications.
    """
    return F.l1_loss(est_wave, ref_wave)


class LossFunction(object):
    """
    CMGAN-Style Loss Function (without discriminator)

    Formula: 0.9 * MSE(Mag) + 0.1 * MSE(R,I) + 0.2 * L1(Time)

    Weight ratios match CMGAN exactly:
    - Magnitude: 0.9 (DOMINANT - 9x more than complex)
    - Complex (R,I): 0.1
    - Time-domain: 0.2

    Key insight: CMGAN emphasizes magnitude reconstruction,
    not complex spectrum. This is the OPPOSITE of the original loss.
    """

    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)

        # CMGAN weights (from actual implementation)
        self.w_mag = 0.9   # Magnitude loss weight (DOMINANT)
        self.w_ri = 0.1    # Real-Imaginary loss weight
        self.w_time = 0.2  # Time-domain loss weight

        print(f"[LossFunction-Ab3] CMGAN-style loss")
        print(f"  Weights: Mag={self.w_mag}, RI={self.w_ri}, Time={self.w_time}")
        print(f"  Ratio Mag:RI = {self.w_mag/self.w_ri:.0f}:1 (magnitude dominant)")

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples):
        # Apply mask
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask

        # 1. Magnitude loss (MSE) - DOMINANT term
        loss_mag = magnitude_mse_loss(est_masked, lbl_masked)

        # 2. Complex component loss (MSE on real/imaginary)
        loss_ri = mse_loss_complex(est_masked, lbl_masked)

        # 3. Time-domain loss (L1 on waveform)
        est_wave = self.resynthesizer(est_masked, mix)
        lbl_wave = self.resynthesizer(lbl_masked, mix)
        T = n_samples[0].item()
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        loss_time = time_domain_l1_loss(est_wave, lbl_wave)

        # Combined loss: CMGAN-style weights
        # 0.9 * MSE(Mag) + 0.1 * MSE(R,I) + 0.2 * L1(Time)
        total_loss = (self.w_mag * loss_mag +
                      self.w_ri * loss_ri +
                      self.w_time * loss_time)

        return total_loss
