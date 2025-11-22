"""
networks_s2_s4.py - Network for experiments S2 and S4
=====================================================
F=201 (win=400, hop=100)

USAGE: cp networks_s2_s4.py utils/networks.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveFrequencyBandPositionalEncoding(nn.Module):
    def __init__(self, strategy='learnable-sin', F=201, sample_rate=16000):
        super().__init__()
        self.strategy = strategy
        self.sample_rate = sample_rate
        self.F_init = F

        self.low_freq_range = (0, 300)
        self.mid_freq_range = (300, 3400)
        self.high_freq_range = (3400, sample_rate / 2)

        nyquist = sample_rate / 2
        self.low_bins_init = int(F * (self.low_freq_range[1] / nyquist))
        self.mid_bins_init = int(F * (self.mid_freq_range[1] / nyquist)) - self.low_bins_init
        self.high_bins_init = F - (self.low_bins_init + self.mid_bins_init)

        self.pe_low = self._init_hybrid_pe(self.low_bins_init, offset=0)
        self.pe_mid = self._init_hybrid_pe(self.mid_bins_init, offset=self.low_bins_init)
        self.pe_high = self._init_hybrid_pe(self.high_bins_init, offset=self.low_bins_init + self.mid_bins_init)

    def _init_hybrid_pe(self, num_bins, offset):
        if num_bins <= 0:
            return None
        position = torch.arange(num_bins).float()
        div_term = torch.exp(torch.arange(0, 1, 2).float() * (-math.log(10000.0)))
        pe_fixed = torch.zeros(num_bins)
        pe_fixed += torch.sin((position + offset) / self.F_init * div_term)
        pe_residual = nn.Parameter(torch.zeros(num_bins), requires_grad=True)
        return nn.Parameter(pe_fixed + pe_residual, requires_grad=True)

    def forward(self, X):
        # X shape: [B, C, T, F] where T=time, F=frequency
        batch_size, C, T, freq_bins = X.size()  # FIXED: was incorrectly (C, freq_bins, T)
        device = X.device
        nyquist = self.sample_rate / 2
        current_low = int(freq_bins * (self.low_freq_range[1] / nyquist))
        current_mid = int(freq_bins * (self.mid_freq_range[1] / nyquist)) - current_low
        current_high = freq_bins - (current_low + current_mid)

        # PE shape: [B, C, T, F] - applied to frequency dimension (dim 3)
        pe_adaptive = torch.zeros(batch_size, C, T, freq_bins, device=device)

        if self.pe_low is not None and current_low > 0:
            pe_low = F.interpolate(self.pe_low.unsqueeze(0).unsqueeze(0), size=current_low, mode='linear', align_corners=False)
            pe_low = pe_low.unsqueeze(2)  # Add time dimension at dim 2
            # FIXED: Apply to dim 3 (frequency), not dim 2 (time)
            pe_adaptive[:, :, :, :current_low] = pe_low.expand(batch_size, C, T, current_low)

        if self.pe_mid is not None and current_mid > 0:
            pe_mid = F.interpolate(self.pe_mid.unsqueeze(0).unsqueeze(0), size=current_mid, mode='linear', align_corners=False)
            pe_mid = pe_mid.unsqueeze(2)
            # FIXED: Apply to dim 3 (frequency), not dim 2 (time)
            pe_adaptive[:, :, :, current_low:current_low + current_mid] = pe_mid.expand(batch_size, C, T, current_mid)

        if self.pe_high is not None and current_high > 0:
            pe_high = F.interpolate(self.pe_high.unsqueeze(0).unsqueeze(0), size=current_high, mode='linear', align_corners=False)
            pe_high = pe_high.unsqueeze(2)
            # FIXED: Apply to dim 3 (frequency), not dim 2 (time)
            pe_adaptive[:, :, :, current_low + current_mid:] = pe_high.expand(batch_size, C, T, current_high)

        return pe_adaptive


class DepthwiseFrequencyAttention(nn.Module):
    """Applies attention along the FREQUENCY dimension (dim 3)."""
    def __init__(self, in_channels, kernel_size=5):
        super().__init__()
        # FIXED: kernel (1, kernel_size) operates on dim 3 (frequency), not dim 2 (time)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), groups=in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, T, F] - attention computed along F (frequency)
        return x * self.sigmoid(self.dw_conv(x))


class GatedPositionalEncoding(nn.Module):
    def __init__(self, in_channels, F=201, sample_rate=16000):
        super().__init__()
        self.positional_encoding = AdaptiveFrequencyBandPositionalEncoding(F=F, sample_rate=sample_rate)
        self.dw_attention = DepthwiseFrequencyAttention(in_channels)
        self.gate = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, X):
        P_freq = self.positional_encoding(X)
        attn = self.dw_attention(X)
        gate = self.gate(X)
        return X + gate * attn * P_freq


class FACLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, F=201, sample_rate=16000):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gated_pe = GatedPositionalEncoding(in_channels, F=F, sample_rate=sample_rate)

    def forward(self, X):
        X = self.gated_pe(X)
        return self.conv(X)


class Net(nn.Module):
    """Network for S2/S4: F=201"""
    def __init__(self, F=201):
        super().__init__()
        self.F = 201  # Hardcoded for S2/S4

        # Encoder
        self.conv1 = FACLayer(2, 16, (2,3), (1,2), (1,0), 201)
        self.conv2 = FACLayer(16, 32, (2,3), (1,2), (1,0), 201)
        self.conv3 = FACLayer(32, 64, (2,3), (1,2), (1,0), 201)
        self.conv4 = FACLayer(64, 128, (2,3), (1,2), (1,0), 201)
        self.conv5 = FACLayer(128, 256, (2,3), (1,2), (1,0), 201)

        # AIA Module
        self.m1 = AIA_Transformer(256, 256)

        # Decoder - F=201: needs output_padding=(0,1) on de4 for skip connection
        self.de5 = nn.ConvTranspose2d(512, 128, (2,3), (1,2), (1,0))
        self.de4 = nn.ConvTranspose2d(256, 64, (2,3), (1,2), (1,0), output_padding=(0,1))  # KEY DIFFERENCE
        self.de3 = nn.ConvTranspose2d(128, 32, (2,3), (1,2), (1,0))
        self.de2 = nn.ConvTranspose2d(64, 16, (2,3), (1,2), (1,0), output_padding=(0,1))
        self.de1 = nn.ConvTranspose2d(32, 2, (2,3), (1,2), (1,0))

        # Instance Norms
        self.bn1 = nn.InstanceNorm2d(16, affine=True)
        self.bn2 = nn.InstanceNorm2d(32, affine=True)
        self.bn3 = nn.InstanceNorm2d(64, affine=True)
        self.bn4 = nn.InstanceNorm2d(128, affine=True)
        self.bn5 = nn.InstanceNorm2d(256, affine=True)
        self.bn5_t = nn.InstanceNorm2d(128, affine=True)
        self.bn4_t = nn.InstanceNorm2d(64, affine=True)
        self.bn3_t = nn.InstanceNorm2d(32, affine=True)
        self.bn2_t = nn.InstanceNorm2d(16, affine=True)
        self.bn1_t = nn.InstanceNorm2d(2, affine=True)

        self.elu = nn.ELU(inplace=True)

    def forward(self, x, global_step=None):
        e1 = self.elu(self.bn1(self.conv1(x)[:,:,:-1]))
        e2 = self.elu(self.bn2(self.conv2(e1)[:,:,:-1]))
        e3 = self.elu(self.bn3(self.conv3(e2)[:,:,:-1]))
        e4 = self.elu(self.bn4(self.conv4(e3)[:,:,:-1]))
        e5 = self.elu(self.bn5(self.conv5(e4)[:,:,:-1]))

        aia_out = self.m1(e5)
        out = torch.cat([aia_out, e5], dim=1)

        d5 = self.elu(self.bn5_t(F.pad(self.de5(out), [0,0,1,0])))
        out = torch.cat([d5, e4], dim=1)
        d4 = self.elu(self.bn4_t(F.pad(self.de4(out), [0,0,1,0])))
        out = torch.cat([d4, e3], dim=1)
        d3 = self.elu(self.bn3_t(F.pad(self.de3(out), [0,0,1,0])))
        out = torch.cat([d3, e2], dim=1)
        d2 = self.elu(self.bn2_t(F.pad(self.de2(out), [0,0,1,0])))
        out = torch.cat([d2, e1], dim=1)
        d1 = self.bn1_t(F.pad(self.de1(out), [0,0,1,0]))
        return d1


class Hybrid_SelfAttention_MRHA3(nn.Module):
    def __init__(self, in_channels, downsample_stride=2):
        super().__init__()
        self.in_channels = in_channels
        self.downsample_stride = downsample_stride
        self.downsample = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=downsample_stride, padding=1)
        self.query_cross = nn.Linear(in_channels, in_channels)
        self.key_cross = nn.Linear(in_channels, in_channels)
        self.value_cross = nn.Linear(in_channels, in_channels)
        self.norm_cross = nn.LayerNorm(in_channels)
        self.query_dot = nn.Linear(in_channels, in_channels)
        self.key_dot = nn.Linear(in_channels, in_channels)
        self.value_dot = nn.Linear(in_channels, in_channels)
        self.norm_dot = nn.LayerNorm(in_channels)
        self.query_cos = nn.Linear(in_channels, in_channels)
        self.key_cos = nn.Linear(in_channels, in_channels)
        self.value_cos = nn.Linear(in_channels, in_channels)
        self.norm_cos = nn.LayerNorm(in_channels)
        self.gate_conv = nn.Conv1d(3 * in_channels, 3, kernel_size=1)
        self.eps = 1e-8

    def forward(self, x):
        B, T, C = x.shape
        x_down = self.downsample(x.permute(0,2,1)).permute(0,2,1)
        q_cross = self.query_cross(x)
        k_cross = self.key_cross(x_down)
        v_cross = self.value_cross(x_down)
        attn_cross = F.softmax(torch.bmm(q_cross, k_cross.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_cross = self.norm_cross(torch.bmm(attn_cross, v_cross))

        q_dot = self.query_dot(x)
        k_dot = self.key_dot(x)
        v_dot = self.value_dot(x)
        attn_dot = F.softmax(torch.bmm(q_dot, k_dot.transpose(1,2)) / math.sqrt(C), dim=-1)
        out_dot = self.norm_dot(torch.bmm(attn_dot, v_dot))

        q_cos = self.query_cos(x)
        k_cos = self.key_cos(x)
        q_norm = q_cos / (q_cos.norm(dim=2, keepdim=True) + self.eps)
        k_norm = k_cos / (k_cos.norm(dim=2, keepdim=True) + self.eps)
        attn_cos = F.softmax(torch.bmm(q_norm, k_norm.transpose(1,2)), dim=-1)
        out_cos = self.norm_cos(torch.bmm(attn_cos, self.value_cos(x)))

        fused = torch.cat([out_cross, out_dot, out_cos], dim=-1).permute(0,2,1)
        gating = F.softmax(self.gate_conv(fused), dim=1)
        gating = gating.permute(0,2,1).unsqueeze(2)
        outputs = torch.stack([out_cross, out_dot, out_cos], dim=3)
        z = torch.sum(gating * outputs, dim=3)
        return z


class AIA_Transformer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input = nn.Sequential(nn.Conv2d(input_size, input_size//2, kernel_size=1), nn.PReLU(), nn.Dropout(dropout))
        self.row_trans = Hybrid_SelfAttention_MRHA3(input_size//2)
        self.col_trans = Hybrid_SelfAttention_MRHA3(input_size//2)
        self.row_norm = nn.InstanceNorm2d(input_size//2, affine=True)
        self.col_norm = nn.InstanceNorm2d(input_size//2, affine=True)
        self.k1 = nn.Parameter(torch.ones(1))
        self.k2 = nn.Parameter(torch.ones(1))
        self.output = nn.Sequential(nn.PReLU(), nn.Conv2d(input_size//2, output_size, kernel_size=1), nn.Dropout(dropout))

    def forward(self, x):
        B, C, d2, d1 = x.shape
        x = self.input(x)
        row_in = x.permute(0,2,3,1).contiguous().view(B*d2, d1, -1)
        row_out = self.row_trans(row_in).view(B, d2, d1, -1).permute(0,3,1,2)
        row_out = self.row_norm(row_out)
        col_in = x.permute(0,3,2,1).contiguous().view(B*d1, d2, -1)
        col_out = self.col_trans(col_in).view(B, d1, d2, -1).permute(0,3,2,1)
        col_out = self.col_norm(col_out)
        out = x + self.k1 * row_out + self.k2 * col_out
        return self.output(out)
