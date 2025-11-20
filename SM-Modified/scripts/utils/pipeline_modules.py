import torch
import torch.nn.functional as F
from utils.stft import STFT


class NetFeeder(object):
    """
    Converts time-domain signals to frequency-domain features.
    
    Optimizations:
    - Removed unused epsilon (code clarity)
    - Added docstrings
    """
    def __init__(self, device, win_size=320, hop_size=160):
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, mix, sph):
        """
        Convert time-domain mix and speech to complex spectra.
        
        Args:
            mix: Mixed signal [batch, samples]
            sph: Clean speech [batch, samples]
            
        Returns:
            feat: Mix spectrum [batch, 2, frames, freq_bins]
            lbl: Speech spectrum [batch, 2, frames, freq_bins]
        """
        real_mix, imag_mix = self.stft.stft(mix)
        feat = torch.stack([real_mix, imag_mix], dim=1)
        
        real_sph, imag_sph = self.stft.stft(sph)
        lbl = torch.stack([real_sph, imag_sph], dim=1)
        
        return feat, lbl


class Resynthesizer(object):
    """
    Converts frequency-domain estimates back to time-domain.
    
    Optimizations:
    - Safe length handling (prevents crashes)
    - Handles both shorter and longer iSTFT outputs
    """
    def __init__(self, device, win_size=320, hop_size=160):
        self.stft = STFT(win_size, hop_size).to(device)

    def __call__(self, est, mix):
        """
        Convert estimated spectrum to time-domain waveform.
        
        Args:
            est: Estimated spectrum [batch, 2, frames, freq_bins]
            mix: Original mix (for length reference) [batch, samples]
            
        Returns:
            sph_est: Estimated speech [batch, samples]
        """
        # Inverse STFT
        sph_est = self.stft.istft(est)
        
        # OPTIMIZATION: Safe length matching
        target_len = mix.shape[1]
        current_len = sph_est.shape[1]
        
        if current_len < target_len:
            # Pad if too short
            sph_est = F.pad(sph_est, [0, target_len - current_len])
        elif current_len > target_len:
            # Truncate if too long
            sph_est = sph_est[:, :target_len]
        
        return sph_est