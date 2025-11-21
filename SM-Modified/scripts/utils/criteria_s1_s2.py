"""
criteria_s1_s2.py - Loss function for experiments S1 and S2
===========================================================
Standard Magnitude Loss (no compression)

Loss: SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Magnitude)

USAGE: cp criteria_s1_s2.py utils/criteria.py
"""

import torch
import torch.nn.functional as F
from utils.pipeline_modules import Resynthesizer


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


def magnitude_mse_loss(est, ref, eps=1e-8):
    """Standard MSE loss on magnitude spectrum (no compression)."""
    est_mag = torch.sqrt(est[:, 0]**2 + est[:, 1]**2 + eps)
    ref_mag = torch.sqrt(ref[:, 0]**2 + ref[:, 1]**2 + eps)
    return F.mse_loss(est_mag, ref_mag)


class LossFunction(object):
    """
    Loss for S1/S2: SI-SDR + 10.0*L1 + 3.0*MSE(Mag)
    Standard magnitude loss (no compression)
    """

    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)
        print("[LossFunction] Using STANDARD magnitude loss")

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples):
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask

        # Time-domain loss (SI-SDR)
        est_wave = self.resynthesizer(est_masked, mix)
        lbl_wave = self.resynthesizer(lbl_masked, mix)
        T = n_samples[0].item()
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        loss_sisdr = si_sdr_loss(est_wave, lbl_wave)

        # Complex L1 loss
        loss_mae = l1_loss_complex(est_masked, lbl_masked)

        # Standard magnitude MSE loss
        loss_mag = magnitude_mse_loss(est_masked, lbl_masked)

        # Combined: SI-SDR + 10.0*L1 + 3.0*MSE(Mag)
        return loss_sisdr + 10.0 * loss_mae + 3.0 * loss_mag
