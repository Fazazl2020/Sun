"""
networks_ab6_basic.py - Transformer Basic Fix
==============================================
Ab6-basic = Ab5 (FAC Fix) + Transformer Basic Fixes

Transformer Fixes (minimal, ~0.1% overhead):
1. Cosine temperature: Learnable temperature for sharper attention
2. Transformer PE: Time and frequency positional encoding

Ab5 Changes (kept):
- FAC: Multi-scale PE, learnable band weights, adaptive scaling
- Encoder: kernel (2,5), padding (1,1), PReLU
- No InstanceNorm on final output

Expected improvement: +0.05 to +0.13 PESQ
Parameter overhead: ~550 params (~0.1%)
MAC overhead: ~0%

Configuration: F=201 (win=400, hop=100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveFrequencyBandPositionalEncoding(nn.Module):
    """
    FIXED: Learnable positional encoding for different frequency bands.
    (Same as Ab5)
    """
    def __init__(self, F=201, sample_rate=16000, d_pe=16):
        super().__init__()
        self.sample_rate = sample_rate
        self.F_init = F
        self.d_pe = d_pe

        self.low_freq_range = (0, 300)
        self.mid_freq_range = (300, 3400)
        self.high_freq_range = (3400, sample_rate / 2)

        nyquist = sample_rate / 2
        self.low_bins_init = int(F * (self.low_freq_range[1] / nyquist))
        self.mid_bins_init = int(F * (self.mid_freq_range[1] / nyquist)) - self.low_bins_init
        self.high_bins_init = F - (self.low_bins_init + self.mid_bins_init)

        self.pe_low = self._init_multiscale_pe(self.low_bins_init, offset=0)
        self.pe_mid = self._init_multiscale_pe(self.mid_bins_init, offset=self.low_bins_init)
        self.pe_high = self._init_multiscale_pe(self.high_bins_init, offset=self.low_bins_init + self.mid_bins_init)

        self.band_weights = nn.Parameter(torch.tensor([0.5, 1.0, 0.3]))

    def _init_multiscale_pe(self, num_bins, offset):
        if num_bins <= 0:
            return None

        position = torch.arange(num_bins).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_pe, 2).float() * (-math.log(10000.0) / self.d_pe)
        )

        pe = torch.zeros(num_bins, self.d_pe)
        pe[:, 0::2] = torch.sin((position + offset) * div_term)
        pe[:, 1::2] = torch.cos((position + offset) * div_term)

        pe_combined = pe.mean(dim=1)
        pe_combined = (pe_combined - pe_combined.min()) / (pe_combined.max() - pe_combined.min() + 1e-8)

        return nn.Parameter(pe_combined, requires_grad=True)

    def forward(self, X):
        batch_size, C, T, freq_bins = X.size()
        device = X.device
        nyquist = self.sample_rate / 2

        current_low = int(freq_bins * (self.low_freq_range[1] / nyquist))
        current_mid = int(freq_bins * (self.mid_freq_range[1] / nyquist)) - current_low
        current_high = freq_bins - (current_low + current_mid)

        pe_adaptive = torch.zeros(batch_size, C, T, freq_bins, device=device)
        band_weights = F.softmax(self.band_weights, dim=0) * 3.0

        if self.pe_low is not None and current_low > 0:
            pe_low = F.interpolate(
                self.pe_low.unsqueeze(0).unsqueeze(0),
                size=current_low, mode='linear', align_corners=False
            )
            pe_low = pe_low.unsqueeze(2) * band_weights[0]
            pe_adaptive[:, :, :, :current_low] = pe_low.expand(batch_size, C, T, current_low)

        if self.pe_mid is not None and current_mid > 0:
            pe_mid = F.interpolate(
                self.pe_mid.unsqueeze(0).unsqueeze(0),
                size=current_mid, mode='linear', align_corners=False
            )
            pe_mid = pe_mid.unsqueeze(2) * band_weights[1]
            pe_adaptive[:, :, :, current_low:current_low + current_mid] = pe_mid.expand(batch_size, C, T, current_mid)

        if self.pe_high is not None and current_high > 0:
            pe_high = F.interpolate(
                self.pe_high.unsqueeze(0).unsqueeze(0),
                size=current_high, mode='linear', align_corners=False
            )
            pe_high = pe_high.unsqueeze(2) * band_weights[2]
            pe_adaptive[:, :, :, current_low + current_mid:] = pe_high.expand(batch_size, C, T, current_high)

        return pe_adaptive


class DepthwiseFrequencyAttention(nn.Module):
    """Applies attention along the FREQUENCY dimension (dim 3)."""
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels,
                                 kernel_size=(1, kernel_size),
                                 padding=(0, kernel_size//2),
                                 groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.dw_conv(x))


class GatedPositionalEncoding(nn.Module):
    """Gated PE with adaptive scaling (same as Ab5)"""
    def __init__(self, in_channels, F=201, sample_rate=16000):
        super().__init__()
        self.positional_encoding = AdaptiveFrequencyBandPositionalEncoding(F=F, sample_rate=sample_rate)
        self.dw_attention = DepthwiseFrequencyAttention(in_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, X):
        P_freq = self.positional_encoding(X)
        attn = self.dw_attention(X)
        gate = self.gate(X)

        scale = self.scale_net(X)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        P_freq_scaled = P_freq * scale

        return X + gate * attn * P_freq_scaled


class FACLayer(nn.Module):
    """Frequency-Adaptive Convolution Layer with fixed PE"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, F=201, sample_rate=16000):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gated_pe = GatedPositionalEncoding(in_channels, F=F, sample_rate=sample_rate)

    def forward(self, X):
        X = self.gated_pe(X)
        return self.conv(X)


class Hybrid_SelfAttention_MRHA3(nn.Module):
    """
    Multi-Resolution Hybrid Attention with 3 branches

    FIX 1: Added learnable temperature for cosine branch
    - Cosine similarity gives values in [-1, 1]
    - Without temperature, softmax is too flat
    - Learnable temperature allows model to sharpen attention
    """
    def __init__(self, in_channels, downsample_stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.downsample_stride = downsample_stride

        # Cross-Resolution Branch
        self.downsample = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=downsample_stride, padding=1)
        self.query_cross = nn.Linear(in_channels, in_channels)
        self.key_cross = nn.Linear(in_channels, in_channels)
        self.value_cross = nn.Linear(in_channels, in_channels)
        self.norm_cross = nn.LayerNorm(in_channels)

        # Local Dot Branch
        self.query_dot = nn.Linear(in_channels, in_channels)
        self.key_dot = nn.Linear(in_channels, in_channels)
        self.value_dot = nn.Linear(in_channels, in_channels)
        self.norm_dot = nn.LayerNorm(in_channels)

        # Local Cosine Branch
        self.query_cos = nn.Linear(in_channels, in_channels)
        self.key_cos = nn.Linear(in_channels, in_channels)
        self.value_cos = nn.Linear(in_channels, in_channels)
        self.norm_cos = nn.LayerNorm(in_channels)

        # ===========================================
        # FIX 1: Learnable temperature for cosine attention
        # ===========================================
        # Initialize to 0.1 (will give 10x sharper attention than default)
        # Learnable so model can adjust during training
        self.cos_temperature = nn.Parameter(torch.tensor(0.1))

        # 3-Way Gating
        self.gate_conv = nn.Conv1d(3 * in_channels, 3, kernel_size=1)
        self.eps = 1e-8

    def forward(self, x):
        B, T, C = x.shape

        # Cross-Resolution Branch (unchanged)
        x_down = self.downsample(x.permute(0,2,1)).permute(0,2,1)
        q_cross = self.query_cross(x)
        k_cross = self.key_cross(x_down)
        v_cross = self.value_cross(x_down)
        attn_cross = F.softmax(torch.bmm(q_cross, k_cross.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_cross = self.norm_cross(torch.bmm(attn_cross, v_cross))

        # Local Dot Branch (unchanged)
        q_dot = self.query_dot(x)
        k_dot = self.key_dot(x)
        v_dot = self.value_dot(x)
        attn_dot = F.softmax(torch.bmm(q_dot, k_dot.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_dot = self.norm_dot(torch.bmm(attn_dot, v_dot))

        # Local Cosine Branch - WITH TEMPERATURE FIX
        q_cos = self.query_cos(x)
        k_cos = self.key_cos(x)
        q_norm = q_cos / (q_cos.norm(dim=2, keepdim=True) + self.eps)
        k_norm = k_cos / (k_cos.norm(dim=2, keepdim=True) + self.eps)

        # FIX: Divide by learnable temperature for sharper attention
        # clamp temperature to avoid division by zero or negative
        temperature = self.cos_temperature.clamp(min=0.01)
        attn_cos = F.softmax(torch.bmm(q_norm, k_norm.transpose(1,2)) / temperature, dim=-1)
        out_cos = self.norm_cos(torch.bmm(attn_cos, self.value_cos(x)))

        # 3-Way Gating (unchanged)
        fused = torch.cat([out_cross, out_dot, out_cos], dim=-1).permute(0,2,1)
        gating = F.softmax(self.gate_conv(fused), dim=1)
        gating = gating.permute(0,2,1).unsqueeze(2)

        outputs = torch.stack([out_cross, out_dot, out_cos], dim=3)
        z = torch.sum(gating * outputs, dim=3)
        return z


class AIA_Transformer(nn.Module):
    """
    Attention-in-Attention Transformer for T-F processing

    FIX 2: Added time and frequency positional encoding
    - Self-attention is permutation invariant
    - PE helps model understand temporal and frequency structure
    - Uses sinusoidal PE (dynamic, works for any length)
    """
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = input_size // 2

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size//2, kernel_size=1),
            nn.PReLU(),
            nn.Dropout(dropout)
        )

        self.row_trans = Hybrid_SelfAttention_MRHA3(input_size//2)
        self.col_trans = Hybrid_SelfAttention_MRHA3(input_size//2)

        self.row_norm = nn.InstanceNorm2d(input_size//2, affine=True)
        self.col_norm = nn.InstanceNorm2d(input_size//2, affine=True)

        self.k1 = nn.Parameter(torch.ones(1))
        self.k2 = nn.Parameter(torch.ones(1))

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(input_size//2, output_size, kernel_size=1),
            nn.Dropout(dropout)
        )

        # ===========================================
        # FIX 2: Learnable scale for sinusoidal PE
        # ===========================================
        # Instead of fixed-size learnable PE, use sinusoidal PE with learnable scale
        # This works for ANY input length
        self.pe_scale = nn.Parameter(torch.tensor(0.1))  # Learnable scale

    def _get_sinusoidal_pe(self, length, channels, device):
        """Generate sinusoidal positional encoding for any length"""
        position = torch.arange(length, device=device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2, device=device).float() * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(length, channels, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        B, C, d2, d1 = x.shape  # d2=time, d1=freq
        x = self.input(x)
        _, C_hidden, _, _ = x.shape

        # FIX 2: Dynamic sinusoidal PE (works for any length)
        time_pe = self._get_sinusoidal_pe(d2, C_hidden, x.device)  # [d2, C]
        freq_pe = self._get_sinusoidal_pe(d1, C_hidden, x.device)  # [d1, C]

        # Reshape and add: time_pe [1, C, d2, 1], freq_pe [1, C, 1, d1]
        time_pe = time_pe.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # [1, C, d2, 1]
        freq_pe = freq_pe.permute(1, 0).unsqueeze(0).unsqueeze(2)   # [1, C, 1, d1]

        x = x + self.pe_scale * (time_pe + freq_pe)

        # Row-wise attention (frequency dimension)
        row_in = x.permute(0,2,3,1).contiguous().view(B*d2, d1, -1)
        row_out = self.row_trans(row_in).view(B, d2, d1, -1).permute(0,3,1,2)
        row_out = self.row_norm(row_out)

        # Column-wise attention (time dimension)
        col_in = x.permute(0,3,2,1).contiguous().view(B*d1, d2, -1)
        col_out = self.col_trans(col_in).view(B, d1, d2, -1).permute(0,3,2,1)
        col_out = self.col_norm(col_out)

        # Fusion
        out = x + self.k1 * row_out + self.k2 * col_out
        return self.output(out)


class Net(nn.Module):
    """
    Speech Enhancement Network - Ab6-basic (Transformer Basic Fix)

    Ab6-basic = Ab5 + Transformer Basic Fixes

    Transformer Fixes:
    1. Cosine temperature: Learnable temperature for sharper attention
    2. Transformer PE: Time and frequency positional encoding

    Ab5 Changes (kept):
    - FAC: Multi-scale PE, learnable band weights, adaptive scaling
    - Encoder: kernel (2,5), padding (1,1), PReLU
    - No InstanceNorm on final output

    Config: F=201 (win=400, hop=100)
    """
    def __init__(self, F=201):
        super().__init__()
        self.F = F

        # ============================================================
        # ENCODER with FIXED FAC (same as Ab5)
        # ============================================================
        self.conv1 = FACLayer(2, 16, (2,5), (1,2), (1,1), F)
        self.conv2 = FACLayer(16, 32, (2,5), (1,2), (1,1), F)
        self.conv3 = FACLayer(32, 64, (2,5), (1,2), (1,1), F)
        self.conv4 = FACLayer(64, 128, (2,5), (1,2), (1,1), F)
        self.conv5 = FACLayer(128, 256, (2,5), (1,2), (1,1), F)

        # AIA Transformer with PE fix
        self.m1 = AIA_Transformer(256, 256)

        # ============================================================
        # DECODER (same as Ab5)
        # ============================================================
        self.de5 = nn.ConvTranspose2d(512, 128, (2,5), (1,2), (1,1))
        self.de4 = nn.ConvTranspose2d(256, 64, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de3 = nn.ConvTranspose2d(128, 32, (2,5), (1,2), (1,1))
        self.de2 = nn.ConvTranspose2d(64, 16, (2,5), (1,2), (1,1), output_padding=(0,1))
        self.de1 = nn.ConvTranspose2d(32, 2, (2,5), (1,2), (1,1))

        # Encoder norms
        self.bn1 = nn.InstanceNorm2d(16, affine=True)
        self.bn2 = nn.InstanceNorm2d(32, affine=True)
        self.bn3 = nn.InstanceNorm2d(64, affine=True)
        self.bn4 = nn.InstanceNorm2d(128, affine=True)
        self.bn5 = nn.InstanceNorm2d(256, affine=True)

        # Decoder norms (except final layer!)
        self.bn5_t = nn.InstanceNorm2d(128, affine=True)
        self.bn4_t = nn.InstanceNorm2d(64, affine=True)
        self.bn3_t = nn.InstanceNorm2d(32, affine=True)
        self.bn2_t = nn.InstanceNorm2d(16, affine=True)

        # PReLU activations (learnable per-channel)
        self.prelu1 = nn.PReLU(16)
        self.prelu2 = nn.PReLU(32)
        self.prelu3 = nn.PReLU(64)
        self.prelu4 = nn.PReLU(128)
        self.prelu5 = nn.PReLU(256)
        self.prelu5_t = nn.PReLU(128)
        self.prelu4_t = nn.PReLU(64)
        self.prelu3_t = nn.PReLU(32)
        self.prelu2_t = nn.PReLU(16)

    def forward(self, x, global_step=None):
        # Encoder with PReLU
        e1 = self.prelu1(self.bn1(self.conv1(x)[:,:,:-1]))
        e2 = self.prelu2(self.bn2(self.conv2(e1)[:,:,:-1]))
        e3 = self.prelu3(self.bn3(self.conv3(e2)[:,:,:-1]))
        e4 = self.prelu4(self.bn4(self.conv4(e3)[:,:,:-1]))
        e5 = self.prelu5(self.bn5(self.conv5(e4)[:,:,:-1]))

        # AIA with transformer fixes
        aia_out = self.m1(e5)
        out = torch.cat([aia_out, e5], dim=1)

        # Decoder with PReLU
        d5 = self.prelu5_t(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))
        out = torch.cat([d5, e4], dim=1)

        d4 = self.prelu4_t(self.bn4_t(F.pad(self.de4(out), [0,0,1,0])))
        out = torch.cat([d4, e3], dim=1)

        d3 = self.prelu3_t(self.bn3_t(F.pad(self.de3(out), [0,0,1,0])))
        out = torch.cat([d3, e2], dim=1)

        d2 = self.prelu2_t(self.bn2_t(F.pad(self.de2(out), [0,0,1,0])))
        out = torch.cat([d2, e1], dim=1)

        # CRITICAL: Final output has NO normalization, NO activation
        d1 = F.pad(self.de1(out), [0,0,1,0])
        return d1
