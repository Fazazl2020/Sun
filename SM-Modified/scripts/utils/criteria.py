"""
Multi-Stage Loss Function
=========================
Supports different experiment configurations:

S1: SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Magnitude)
S2: Same as S1, with F=201 STFT resolution
S3: SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Magnitude^0.3)  [Compressed]
S4: Same as S3, with F=201 STFT resolution

Compressed magnitude loss (mag^0.3) gives more weight to small magnitude
errors, which are perceptually important. This is inspired by CMGAN's
power compression approach.
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


def magnitude_mse_loss(est, ref, eps=1e-8, compress=False, power=0.3):
    """
    MSE loss on magnitude spectrum.

    Args:
        est: Estimated complex spectrum [B, 2, F, T]
        ref: Reference complex spectrum [B, 2, F, T]
        eps: Small value for numerical stability
        compress: If True, apply power compression (mag^power)
        power: Compression power (default 0.3, same as CMGAN)

    Rationale for compression:
        - Raw magnitude has huge dynamic range (e.g., 0.001 to 100)
        - MSE on raw magnitude weights large values heavily
        - Compression (mag^0.3) reduces range: 0.001^0.3=0.06, 100^0.3=4.0
        - This gives more equal weight to small magnitude errors
        - Small magnitudes are perceptually important (quiet sounds)
    """
    est_mag = torch.sqrt(est[:, 0]**2 + est[:, 1]**2 + eps)
    ref_mag = torch.sqrt(ref[:, 0]**2 + ref[:, 1]**2 + eps)

    if compress:
        # Power compression: mag^0.3 (like CMGAN)
        est_mag = est_mag ** power
        ref_mag = ref_mag ** power

    return F.mse_loss(est_mag, ref_mag)


class LossFunction(object):
    """
    Multi-Stage Loss Function

    SI-SDR (time-domain) + L1 (complex) + MSE (magnitude)
    Weights: SI-SDR dominant, L1=10.0, Magnitude=3.0

    Experiment configurations:
    - S1/S2: Standard magnitude loss
    - S3/S4: Compressed magnitude loss (mag^0.3 for perceptual weighting)
    """

    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)

        # Load experiment settings from config
        self.use_compressed_mag = MODEL_CONFIG.get('use_compressed_mag', False)
        self.mag_compression_power = MODEL_CONFIG.get('mag_compression_power', 0.3)

        # Log the configuration
        if self.use_compressed_mag:
            print(f"[LossFunction] Using COMPRESSED magnitude loss (power={self.mag_compression_power})")
        else:
            print(f"[LossFunction] Using standard magnitude loss")

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

        # Magnitude loss (standard or compressed based on experiment)
        loss_mag = magnitude_mse_loss(
            est_masked, lbl_masked,
            compress=self.use_compressed_mag,
            power=self.mag_compression_power
        )

        # Combined loss: SI-SDR dominant + L1 + Magnitude
        # Formula: SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Mag)
        return loss_sisdr + 10.0 * loss_mae + 3.0 * loss_mag