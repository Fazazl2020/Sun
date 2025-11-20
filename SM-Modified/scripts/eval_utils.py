from subprocess import run, PIPE
from scipy.linalg import toeplitz
from scipy.io import wavfile
import numba as nb
from numba import jit, int32, float32
import soundfile as sf
from scipy.signal import lfilter
from scipy.interpolate import interp1d
import torch
import torch.nn.functional as F
import glob
import librosa
import numpy as np
import tempfile
import os
import re
from pesq import pesq


def uttname2spkid(uttname):
    """Extract speaker ID from utterance name"""
    spkid = uttname.split('_')[0]
    return spkid


def denormalize_wave_minmax(x):
    """Denormalize waveform from [-1, 1] to int16 range"""
    return (65535. * x / 2) - 1 + 32767.


def make_divN(tensor, N, method='zeros'):
    """
    Make tensor time dimension divisible by N
    
    Args:
        tensor: Input tensor
        N: Divisor
        method: 'zeros' or 'reflect' padding
    """
    pad_num = (tensor.size(1) + N) - (tensor.size(1) % N) - tensor.size(1)
    if method == 'zeros':
        pad = torch.zeros(tensor.size(0), pad_num, tensor.size(-1))
        return torch.cat((tensor, pad), dim=1)
    elif method == 'reflect':
        tensor = tensor.transpose(1, 2)
        return F.pad(tensor, (0, pad_num), 'reflect').transpose(1, 2)
    else:
        raise TypeError('Unrecognized make_divN pad method: ', method)


def composite_helper(args):
    """Helper for parallel composite evaluation"""
    return eval_composite(*args)


class ComposeAdditive(object):
    """Composition class for additive noise"""
    
    def __init__(self, additive):
        self.additive = additive

    def __call__(self, x):
        return x, self.additive(x)


class Additive(object):
    """
    Additive noise augmentation with SNR control
    Implements ITU-T P.56 active speech level normalization
    """

    def __init__(self, noises_dir, snr_levels=[0, 5, 10], do_IRS=False):
        self.noises_dir = noises_dir
        self.snr_levels = snr_levels
        self.do_IRS = do_IRS
        self.eps = 1e-22
        
        # Read noise files
        noises = glob.glob(os.path.join(noises_dir, '*.wav'))
        if len(noises) == 0:
            raise ValueError('[!] No noises found in {}'.format(noises_dir))
        else:
            print('[*] Found {} noise files'.format(len(noises)))
            self.noises = []
            for n_i, npath in enumerate(noises, start=1):
                nwav = librosa.load(npath, sr=None)[0]
                self.noises.append({'file': npath,
                                    'data': nwav.astype(np.float32)})
                log_noise_load = 'Loaded noise {:3d}/{:3d}: {}'.format(
                    n_i, len(noises), npath)
                print(log_noise_load)

    def __call__(self, wav, srate=16000, nbits=16):
        """
        Add noise to clean waveform
        
        Args:
            wav: Clean waveform
            srate: Sample rate
            nbits: Bit depth
            
        Returns:
            Noisy waveform as torch.FloatTensor
        """
        if isinstance(wav, torch.Tensor):
            wav = wav.numpy()
        
        # FIXED: Replace np.asscalar with .item()
        noise_idx = np.random.choice(list(range(len(self.noises))), 1)
        sel_noise = self.noises[noise_idx.item()]
        noise = sel_noise['data']
        snr = np.random.choice(self.snr_levels, 1)
        
        if wav.ndim > 1:
            wav = wav.reshape((-1,))
        
        noisy, noise_bound = self.addnoise_asl(wav, noise, srate,
                                               nbits, snr,
                                               do_IRS=self.do_IRS)
        
        # Normalize to avoid clipping
        if np.max(noisy) >= 1 or np.min(noisy) < -1:
            small = 0.1
            while np.max(noisy) >= 1 or np.min(noisy) < -1:
                noisy = noisy / (1. + small)
                small = small + 0.1
        
        return torch.FloatTensor(noisy.astype(np.float32))

    def addnoise_asl(self, clean, noise, srate, nbits, snr, do_IRS=False):
        """
        Add noise using active speech level (ASL) normalization
        ITU-T P.56 Method B
        """
        if do_IRS:
            clean = self.apply_IRS(clean, srate, nbits)
        
        Px, asl, c0 = self.asl_P56(clean, srate, nbits)
        x = clean
        x_len = x.shape[0]

        noise_len = noise.shape[0]
        if noise_len <= x_len:
            raise ValueError(f'Noise length ({noise_len}) must be greater than speech length ({x_len})!')
        
        rand_start_limit = int(noise_len - x_len + 1)
        rand_start = int(np.round((rand_start_limit - 1) * np.random.rand(1) + 1))
        noise_segment = noise[rand_start:rand_start + x_len]
        noise_bounds = (rand_start, rand_start + x_len)

        if do_IRS:
            noise_segment = self.apply_IRS(noise_segment, srate, nbits)

        Pn = np.dot(noise_segment.T, noise_segment) / x_len

        # Scale noise to achieve desired SNR
        sf = np.sqrt(Px / (Pn + self.eps) / (10 ** (snr / 10) + self.eps))
        noise_segment = noise_segment * sf

        noisy = x + noise_segment

        return noisy, noise_bounds

    def apply_IRS(self, data, srate, nbits):
        """Apply IRS filter (telephone handset BW [300, 3200] Hz)"""
        raise NotImplementedError('IRS filter under construction!')

    def asl_P56(self, x, srate, nbits):
        """
        ITU-T P.56 Method B - Active Speech Level
        
        Returns:
            asl_ms: Active speech level (mean square)
            asl: Activity factor
            c0: Threshold
        """
        T = 0.03  # Time constant (seconds)
        H = 0.2   # Hangover time (seconds)
        M = 15.9  # Margin (dB)

        thres_no = nbits - 1
        I = np.ceil(srate * H)
        g = np.exp(-1 / (srate * T))
        c = 2. ** (np.array(list(range(-15, (thres_no + 1) - 16))))
        a = np.zeros(c.shape[0])
        hang = np.ones(c.shape[0]) * I

        assert x.ndim == 1, x.shape
        sq = np.dot(x, x)
        x_len = x.shape[0]

        # Envelope detection using 2nd order IIR
        x_abs = np.abs(x)
        p = lfilter(np.ones(1) - g, np.array([1, -g]), x_abs)
        q = lfilter(np.ones(1) - g, np.array([1, -g]), p)

        for k in range(x_len):
            for j in range(thres_no):
                if q[k] >= c[j]:
                    a[j] = a[j] + 1
                    hang[j] = 0
                elif hang[j] < I:
                    a[j] = a[j] + 1
                    hang[j] = hang[j] + 1
                else:
                    break
        
        asl = 0
        asl_ms = 0
        c0 = None
        
        if a[0] == 0:
            return asl_ms, asl, c0
        
        AdB1 = 10 * np.log10(sq / (a[0] + self.eps) + self.eps)
        CdB1 = 20 * np.log10(c[0] + self.eps)
        
        if AdB1 - CdB1 < M:
            return asl_ms, asl, c0
        
        AdB = np.zeros(c.shape[0])
        CdB = np.zeros(c.shape[0])
        Delta = np.zeros(c.shape[0])
        AdB[0] = AdB1
        CdB[0] = CdB1
        Delta[0] = AdB1 - CdB1

        for j in range(1, AdB.shape[0]):
            AdB[j] = 10 * np.log10(sq / (a[j] + self.eps) + self.eps)
            CdB[j] = 20 * np.log10(c[j] + self.eps)

        for j in range(1, Delta.shape[0]):
            if a[j] != 0:
                Delta[j] = AdB[j] - CdB[j]
                if Delta[j] <= M:
                    asl_ms_log, cl0 = self.bin_interp(AdB[j], AdB[j - 1],
                                                       CdB[j], CdB[j - 1],
                                                       M, 0.5)
                    asl_ms = 10 ** (asl_ms_log / 10)
                    asl = (sq / x_len) / (asl_ms + self.eps)
                    c0 = 10 ** (cl0 / 20)
                    break
        
        return asl_ms, asl, c0

    def bin_interp(self, upcount, lwcount, upthr, lwthr, Margin, tol):
        """Binary interpolation for ASL calculation"""
        if tol < 0:
            tol = -tol

        iterno = 1
        if np.abs(upcount - upthr - Margin) < tol:
            return lwcount, lwthr
        if np.abs(lwcount - lwthr - Margin) < tol:
            return lwcount, lwthr

        midcount = (upcount + lwcount) / 2
        midthr = (upthr + lwthr) / 2
        
        while True:
            diff = midcount - midthr - Margin
            if np.abs(diff) <= tol:
                break
            
            iterno += 1
            if iterno > 20:
                tol *= 1.1

            if diff > tol:
                midcount = (upcount + midcount) / 2
                midthr = (upthr + midthr) / 2
            elif diff < -tol:
                midcount = (midcount + lwcount) / 2
                midthr = (midthr + lwthr) / 2
        
        return midcount, midthr


def eval_composite(clean_utt, Genh_utt, noisy_utt=None, srate=16000):
    """
    Evaluate composite metrics (CSIG, CBAK, COVL, PESQ, SSNR)
    
    Args:
        clean_utt: Clean reference signal
        Genh_utt: Enhanced/processed signal
        noisy_utt: Optional noisy signal for comparison
        srate: Sample rate (default 16000)
        
    Returns:
        Dictionary of metrics
    """
    clean_utt = clean_utt.reshape(-1)
    Genh_utt = Genh_utt.reshape(-1)
    
    csig, cbak, covl, pesq_val, ssnr = CompositeEval(clean_utt, Genh_utt, 
                                                       srate=srate, log_all=True)
    evals = {'csig': csig, 'cbak': cbak, 'covl': covl,
             'pesq': pesq_val, 'ssnr': ssnr}
    
    if noisy_utt is not None:
        noisy_utt = noisy_utt.reshape(-1)
        csig, cbak, covl, pesq_val, ssnr = CompositeEval(clean_utt, noisy_utt,
                                                           srate=srate, log_all=True)
        return evals, {'csig': csig, 'cbak': cbak, 'covl': covl,
                       'pesq': pesq_val, 'ssnr': ssnr}
    else:
        return evals


def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """
    Segmental Signal-to-Noise Ratio
    
    Args:
        ref_wav: Reference (clean) signal
        deg_wav: Degraded (processed) signal
        srate: Sample rate
        eps: Small constant for numerical stability
        
    Returns:
        overall_snr: Overall SNR
        segmental_snr: List of per-frame segmental SNR values
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + eps))

    # Frame parameters
    winlength = int(np.round(30 * srate / 1000))  # 30 ms
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        
        frame_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)
        frame_snr = max(frame_snr, MIN_SNR)
        frame_snr = min(frame_snr, MAX_SNR)
        segmental_snr.append(frame_snr)
        
        start += int(skiprate)
    
    return overall_snr, segmental_snr


def CompositeEval(ref_wav, deg_wav, srate=16000, log_all=False):
    """
    Composite metrics: CSIG, CBAK, COVL (Hu & Loizou, 2008)
    
    FIXED: Made sample rate configurable
    
    Args:
        ref_wav: Reference (clean) signal
        deg_wav: Degraded (processed) signal
        srate: Sample rate (default 16000)
        log_all: If True, return all metrics including PESQ and SSNR
        
    Returns:
        Csig, Cbak, Covl [, pesq_raw, segSNR] if log_all=True
    """
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, srate)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, srate)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute SSNR
    snr_mean, segsnr_mean = SSNR(ref_wav, deg_wav, srate)
    segSNR = np.mean(segsnr_mean)

    # Compute PESQ
    # FIXED: Determine mode based on sample rate
    mode = 'wb' if srate == 16000 else 'nb'
    try:
        pesq_raw = pesq(srate, ref_wav, deg_wav, mode)
    except Exception as e:
        print(f"PESQ calculation failed: {e}")
        pesq_raw = 1.0  # Fallback value

    def trim_mos(val):
        """Trim MOS score to [1, 5] range"""
        return min(max(val, 1), 5)

    # Composite metrics formulas from Hu & Loizou (2008)
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)
    
    if log_all:
        return Csig, Cbak, Covl, pesq_raw, segSNR
    else:
        return Csig, Cbak, Covl


def wss(ref_wav, deg_wav, srate):
    """
    Weighted Spectral Slope (WSS) measure
    
    Args:
        ref_wav: Reference signal
        deg_wav: Degraded signal
        srate: Sample rate
        
    Returns:
        List of per-frame WSS distortion values
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, f"Length mismatch: {clean_length} vs {processed_length}"

    winlength = round(30 * srate / 1000.)  # 30 ms
    skiprate = int(np.floor(winlength / 4))
    max_freq = srate / 2
    num_crit = 25

    n_fft = int(2 ** np.ceil(np.log(2 * winlength) / np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions
    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30,
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93,
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423,
                 153.823, 168.154, 183.457, 199.776, 217.153,
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0]
    min_factor = np.exp(-30. / (2 * 2.303))

    # Create critical band filters
    crit_filter = np.zeros((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    # Compute per-frame WSS
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # Compute power spectra
        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        
        # Compute filterbank energies
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * crit_filter[i, :])
        
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))

        # Compute spectral slopes
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit - 1]
        processed_slope = processed_energy[1:num_crit] - processed_energy[:num_crit - 1]

        # Find peak locations
        clean_loc_peak = []
        processed_loc_peak = []
        
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])

        # Compute WSS with weighting
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit - 1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit - 1])
        W_clean = Wmax_clean * Wlocmax_clean
        
        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit - 1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit - 1])
        W_processed = Wmax_processed * Wlocmax_processed
        
        W = (W_clean + W_processed) / 2
        
        frame_distortion = np.sum(W * (clean_slope[:num_crit - 1] - processed_slope[:num_crit - 1]) ** 2)
        frame_distortion = frame_distortion / (np.sum(W) + 1e-10)  # ADDED epsilon
        
        distortion.append(frame_distortion)
        start += int(skiprate)
    
    return distortion


def llr(ref_wav, deg_wav, srate):
    """
    Log-Likelihood Ratio (LLR) measure
    
    Args:
        ref_wav: Reference signal
        deg_wav: Degraded signal
        srate: Sample rate
        
    Returns:
        List of per-frame LLR distortion values
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, f"Length mismatch: {clean_length} vs {processed_length}"

    winlength = round(30 * srate / 1000.)
    skiprate = int(np.floor(winlength / 4))
    
    # LPC analysis order
    P = 10 if srate < 10000 else 16

    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []
    eps = 1e-10  # ADDED epsilon

    for frame_count in range(num_frames):
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        
        # Get LPC coefficients
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]
        
        # Compute LLR
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        
        # ADDED epsilon protection
        log_ = np.log(numerator / (denominator + eps) + eps)
        distortion.append(np.squeeze(log_))
        
        start += int(skiprate)
    
    return np.array(distortion)


def lpcoeff(speech_frame, model_order):
    """
    Linear Predictive Coding (LPC) coefficients using Levinson-Durbin
    
    Args:
        speech_frame: Speech frame
        model_order: LPC model order
        
    Returns:
        acorr: Autocorrelation
        refcoeff: Reflection coefficients
        lpparams: LP parameters
    """
    winlength = speech_frame.shape[0]
    eps = 1e-10  # ADDED epsilon
    
    # Compute autocorrelation lags
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))
    
    # Levinson-Durbin recursion
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0] + eps  # ADDED epsilon
    
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        
        rcoeff[i] = (R[i + 1] - sum_term) / (E[i] + eps)  # ADDED epsilon
        a[i] = rcoeff[i]
        
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    
    return acorr, refcoeff, lpparams