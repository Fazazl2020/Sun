import os
import soundfile as sf
import numpy as np
from pystoi import stoi
from pesq import pesq  # FIXED: Changed from pypesq to pesq
from scipy.signal import fftconvolve
from configs import exp_conf
from utils.utils import getLogger

def snr(ref, est, eps=1e-8):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    
    noise = ref - est
    ratio = np.sum(ref**2) / (np.sum(noise**2) + eps)
    return 10 * np.log10(ratio + eps)


def si_sdr(ref, est, eps=1e-8):
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    
    # CRITICAL: Remove mean (zero-mean normalization)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    
    # Compute the scaling factor (projection of est onto ref)
    ref_energy = np.sum(ref ** 2) + eps
    inner_product = np.sum(ref * est)
    scaling_factor = inner_product / ref_energy
    
    # Compute the projection (target source)
    ref_projection = scaling_factor * ref
    
    # Compute the noise (distortion)
    noise = est - ref_projection
    
    # Compute SI-SDR
    signal_energy = np.sum(ref_projection ** 2)
    noise_energy = np.sum(noise ** 2) + eps
    
    si_sdr_value = 10 * np.log10(signal_energy / noise_energy)
    
    return si_sdr_value


class Metric(object):
    def __init__(self, est_path, ckpt_dir, metric):
        self.sample_rate = exp_conf['sample_rate']
        
        self.est_path = est_path
        self.metric = metric
        assert self.metric in {'stoi', 'estoi', 'pesq', 'snr', 'si_sdr'}
        
        self.ckpt_dir = ckpt_dir
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        # Validate sample rate
        assert self.sample_rate in [8000, 16000], \
            f"Sample rate must be 8000 or 16000, got {self.sample_rate}"

    def evaluate(self):
        getattr(self, self.metric)()

    def apply_metric(self, metric_func):
        logger = getLogger(os.path.join(self.ckpt_dir, self.metric+'_scores.log'), log_file=True)

        all_scores_dir = os.path.join(self.ckpt_dir, 'all_scores')
        if not os.path.isdir(all_scores_dir):
            os.makedirs(all_scores_dir)
        if not os.path.isdir(os.path.join(all_scores_dir, 'scores_arrays')):
            os.makedirs(os.path.join(all_scores_dir, 'scores_arrays'))

        for condition in os.listdir(self.est_path):
            condition_path = os.path.join(self.est_path, condition)
            
            # Skip if not a directory
            if not os.path.isdir(condition_path):
                continue
            
            mix_scores_array = []
            est_scores_array = []

            score_name = condition + '_' + self.metric
            f = open(os.path.join(all_scores_dir, score_name + '.txt'), 'w')
            count = 0
            skipped = 0
            
            for filename in sorted(os.listdir(condition_path)):
                if not filename.endswith('_mix.wav'):
                    continue
                
                try:
                    # Read audio files
                    mix_path = os.path.join(condition_path, filename)
                    sph_path = os.path.join(condition_path, filename.replace('_mix', '_sph'))
                    sph_est_path = os.path.join(condition_path, filename.replace('_mix', '_sph_est'))
                    
                    # Check if all files exist
                    if not os.path.exists(sph_path):
                        logger.warning(f"Clean file not found: {sph_path}")
                        skipped += 1
                        continue
                    if not os.path.exists(sph_est_path):
                        logger.warning(f"Enhanced file not found: {sph_est_path}")
                        skipped += 1
                        continue
                    
                    # Read audio with explicit dtype
                    mix, sr_mix = sf.read(mix_path, dtype='float32')
                    sph, sr_sph = sf.read(sph_path, dtype='float32')
                    sph_est, sr_est = sf.read(sph_est_path, dtype='float32')
                    
                    # Validate sample rates match
                    assert sr_mix == sr_sph == sr_est == self.sample_rate, \
                        f"Sample rate mismatch in {filename}: {sr_mix}, {sr_sph}, {sr_est}"
                    
                    # CRITICAL: Align signal lengths (trim to shortest)
                    min_len = min(len(mix), len(sph), len(sph_est))
                    mix = mix[:min_len]
                    sph = sph[:min_len]
                    sph_est = sph_est[:min_len]
                    
                    # Validate signals are not empty
                    if min_len == 0:
                        logger.warning(f"Empty audio file: {filename}")
                        skipped += 1
                        continue
                    
                    # Validate signals are not all zeros
                    if np.allclose(sph, 0) or np.allclose(mix, 0) or np.allclose(sph_est, 0):
                        logger.warning(f"All-zero audio detected: {filename}")
                        skipped += 1
                        continue
                    
                    # Compute metrics (ref=clean, est=noisy/enhanced)
                    mix_score = metric_func(sph, mix)
                    est_score = metric_func(sph, sph_est)
                    
                    # Validate scores are finite
                    if not (np.isfinite(mix_score) and np.isfinite(est_score)):
                        logger.warning(f"Non-finite score for {filename}: mix={mix_score}, est={est_score}")
                        skipped += 1
                        continue
                    
                    count += 1
                    f.write('utt {}: mix {:.4f}, est {:.4f}\n'.format(filename, mix_score, est_score))
                    f.flush()
                    mix_scores_array.append(mix_score)
                    est_scores_array.append(est_score)
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
                    skipped += 1
                    continue

            if count == 0:
                logger.warning(f"No valid files processed for condition: {condition}")
                f.close()
                continue
            
            mix_scores_array = np.array(mix_scores_array, dtype=np.float32)
            est_scores_array = np.array(est_scores_array, dtype=np.float32)
            
            f.write('========================================\n')
            f.write('{} results: ({} utts, {} skipped)\n'.format(self.metric, count, skipped))
            f.write('mix : {:.4f} +- {:.4f}\n'.format(np.mean(mix_scores_array), np.std(mix_scores_array)))
            f.write('est : {:.4f} +- {:.4f}\n'.format(np.mean(est_scores_array), np.std(est_scores_array)))
            f.close()
            
            np.save(os.path.join(all_scores_dir, 'scores_arrays', score_name + '_mix.npy'), mix_scores_array)
            np.save(os.path.join(all_scores_dir, 'scores_arrays', score_name + '_est.npy'), est_scores_array)

            message = 'Evaluating {}: {} utts: '.format(condition, count) + \
                '{} [ mix: {:.4f}, est: {:.4f} | delta: {:.4f} ]'.format(self.metric, 
                np.mean(mix_scores_array), np.mean(est_scores_array), 
                np.mean(est_scores_array)-np.mean(mix_scores_array))
            logger.info(message)

    def stoi(self):
        """Short-Time Objective Intelligibility"""
        fn = lambda ref, est: stoi(ref, est, self.sample_rate, extended=False)
        self.apply_metric(fn)

    def estoi(self):
        """Extended Short-Time Objective Intelligibility"""
        fn = lambda ref, est: stoi(ref, est, self.sample_rate, extended=True)
        self.apply_metric(fn)
    
    def pesq(self):
        """
        Perceptual Evaluation of Speech Quality
        FIXED: Using correct library and parameter order
        """
        # Determine mode based on sample rate
        mode = 'wb' if self.sample_rate == 16000 else 'nb'
        
        def pesq_func(ref, est):
            """
            Standard PESQ calculation
            Parameter order: pesq(fs, ref, deg, mode)
            """
            # Ensure same length
            min_len = min(len(ref), len(est))
            ref = ref[:min_len]
            est = est[:min_len]
            
            # Call PESQ with correct parameter order
            # fs (sample rate) comes FIRST
            return pesq(self.sample_rate, ref, est, mode)
        
        self.apply_metric(pesq_func)
    
    def snr(self):
        """Signal-to-Noise Ratio"""
        fn = lambda ref, est: snr(ref, est)
        self.apply_metric(fn)
    
    def si_sdr(self):
        """Scale-Invariant Signal-to-Distortion Ratio"""
        fn = lambda ref, est: si_sdr(ref, est)
        self.apply_metric(fn) 