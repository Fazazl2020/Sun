"""
Fixed Weights Loss Function
============================
Ablation Study #1: Weight Balance Fix

Loss: SI-SDR + 10.0 * L1(R,I)

Expected Performance: PESQ 2.90-3.10 (+0.10 from baseline)
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


class LossFunction(object):
    """
    Fixed Weights Loss Function
    
    Weight increased from 0.5 to 10.0 to balance gradient contributions.
    """
    
    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples):
        # Apply mask
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask
        
        # Time-domain loss
        est_wave = self.resynthesizer(est_masked, mix)
        lbl_wave = self.resynthesizer(lbl_masked, mix)
        T = n_samples[0].item()
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        loss_sisdr = si_sdr_loss(est_wave, lbl_wave)
        
        # Complex component loss
        loss_mae = l1_loss_complex(est_masked, lbl_masked)
        
        # Fixed weight: 10.0 instead of 0.5
        return loss_sisdr + 10.0 * loss_mae