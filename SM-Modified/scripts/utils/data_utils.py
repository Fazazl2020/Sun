"""
data_utils.py - OPTIMAL VERSION with PADDING FIX
Version: 5.1 - Fixed for proper evaluation without padding artifacts

Key fixes:
1. Cap file lengths at max_length_samples in cache (prevents invalid segments)
2. NO normalization (preserves SNR - standard practice)
3. Fast soundfile loading
4. Multi-worker support
5. Pin memory for GPU
6. Smart caching with validation
7. Robust error handling

This version ensures fair evaluation by preventing padding artifacts in metrics.
"""

import os
import pickle
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf


def check_persistent_workers_support():
    """Check if PyTorch version supports persistent_workers"""
    try:
        version = torch.__version__.split('+')[0]
        major, minor, _ = map(int, version.split('.'))
        return (major > 1) or (major == 1 and minor >= 7)
    except:
        return False


PERSISTENT_WORKERS_SUPPORTED = check_persistent_workers_support()


class SpeechEnhancementDataset(Dataset):
    """
    Optimal WAV file loader for Speech Enhancement with PADDING FIX
    
    BEST PRACTICES:
    - NO normalization (preserves SNR)
    - Fast soundfile loading
    - Smart error handling
    - Efficient caching
    - FIXED: Caps lengths at max_length_samples to prevent invalid segments
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        sample_rate: int = 16000,
        max_length_seconds: float = 6.0,
        mode: str = 'train',
        random_crop: bool = True,
        cache_in_memory: bool = False,
        length_cache_file: Optional[str] = None,
        skip_on_error: bool = True
    ):
        """
        Args:
            clean_dir: Directory with clean WAV files
            noisy_dir: Directory with noisy WAV files
            sample_rate: Target sample rate (must match files)
            max_length_seconds: Maximum utterance length
            mode: 'train' or 'eval'
            random_crop: Random crop in train mode
            cache_in_memory: Cache loaded audio (good for small val/test sets)
            length_cache_file: Path to cache file lengths
            skip_on_error: Skip bad files instead of crashing
        
        NOTE: NO normalization is applied (best practice for speech enhancement)
        """
        assert mode in ['train', 'eval'], f"Mode must be 'train' or 'eval', got {mode}"
        
        # Validate directories
        if not os.path.isdir(clean_dir):
            raise ValueError(f"Clean directory does not exist: {clean_dir}")
        if not os.path.isdir(noisy_dir):
            raise ValueError(f"Noisy directory does not exist: {noisy_dir}")
        
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate
        self.max_length_samples = int(max_length_seconds * sample_rate)
        self.mode = mode
        self.random_crop = random_crop
        self.cache_in_memory = cache_in_memory
        self.skip_on_error = skip_on_error
        
        # Get sorted file list (reproducibility)
        self.file_list = sorted([
            f for f in os.listdir(clean_dir) 
            if f.endswith('.wav')
        ])
        
        if len(self.file_list) == 0:
            raise ValueError(f"No WAV files found in {clean_dir}")
        
        # Check noisy files exist
        noisy_files = set(os.listdir(noisy_dir))
        missing_files = [f for f in self.file_list if f not in noisy_files]
        
        if missing_files:
            if len(missing_files) > 10:
                warnings.warn(f"{len(missing_files)} files missing in noisy directory")
            else:
                warnings.warn(f"Missing files: {missing_files}")
            
            self.file_list = [f for f in self.file_list if f in noisy_files]
            
            if len(self.file_list) == 0:
                raise ValueError(f"No matching clean/noisy pairs found!")
        
        # Get or compute file lengths (WITH FIX)
        self.lengths = self._load_or_compute_lengths(length_cache_file)
        
        # In-memory cache
        self.memory_cache = {} if cache_in_memory else None
        
        # Track errors
        self.error_count = 0
        
        # Statistics for logging
        self.n_cropped = sum(1 for length in self.lengths if length >= self.max_length_samples)
        self.n_short = len(self.lengths) - self.n_cropped
        
        print(f"\n{'='*60}")
        print(f"[{mode.upper()}] Dataset Initialized (WITH PADDING FIX)")
        print(f"{'='*60}")
        print(f"Files: {len(self.file_list)}")
        print(f"  Short files (= {max_length_seconds}s): {self.n_short} ({100*self.n_short/len(self.file_list):.1f}%)")
        print(f"  Long files (> {max_length_seconds}s): {self.n_cropped} ({100*self.n_cropped/len(self.file_list):.1f}%)")
        if self.n_cropped > 0:
            print(f"  ? Long files will be cropped to {max_length_seconds}s")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Max length: {max_length_seconds}s ({self.max_length_samples} samples)")
        print(f"Normalization: None (preserves SNR - best practice)")
        print(f"Random crop: {random_crop if mode == 'train' else 'N/A'}")
        print(f"Memory cache: {cache_in_memory}")
        print(f"{'='*60}\n")
    
    def _load_or_compute_lengths(self, cache_file: Optional[str]) -> List[int]:
        """
        Load lengths from cache or compute them
        
        ? CRITICAL FIX: Caps lengths at max_length_samples to match actual loaded data
        This prevents SegmentDataset from creating invalid segments beyond cropped length
        """
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    
                    # ? Validate cache (check if max_length_samples changed)
                    if (cached_data['files'] == self.file_list and 
                        cached_data.get('max_length_samples') == self.max_length_samples):
                        print(f"? Loaded lengths from cache: {cache_file}")
                        return cached_data['lengths']
                    else:
                        print(f"??  Cache invalid (files or max_length changed), recomputing...")
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}, recomputing...")
        
        print("Computing file lengths (this may take a minute)...")
        lengths = []
        errors = 0
        
        for i, filename in enumerate(self.file_list):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(self.file_list)} files...")
            
            clean_path = os.path.join(self.clean_dir, filename)
            try:
                info = sf.info(clean_path)
                
                if info.samplerate != self.sample_rate:
                    warnings.warn(
                        f"{filename} has sample rate {info.samplerate}, "
                        f"expected {self.sample_rate}"
                    )
                
                # ??? CRITICAL FIX: Cap at max_length_samples
                # This ensures lengths match what will actually be loaded
                # Prevents SegmentDataset from creating segments beyond cropped data
                actual_length = min(info.frames, self.max_length_samples)
                lengths.append(actual_length)
                
            except Exception as e:
                warnings.warn(f"ERROR reading {filename}: {e}")
                lengths.append(0)
                errors += 1
        
        print(f"  Completed! Processed {len(self.file_list)} files")
        
        if errors > 0:
            warnings.warn(f"{errors} files had errors")
        
        # Save to cache with validation metadata
        if cache_file:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                cache_data = {
                    'files': self.file_list,
                    'lengths': lengths,
                    'max_length_samples': self.max_length_samples  # ? Store for validation
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"? Saved lengths to cache: {cache_file}")
            except Exception as e:
                warnings.warn(f"Failed to save cache: {e}")
        
        return lengths
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one audio pair
        
        BEST PRACTICE: NO normalization - preserves SNR
        """
        # Check cache
        if self.memory_cache is not None and idx in self.memory_cache:
            return self.memory_cache[idx]
        
        filename = self.file_list[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)
        
        try:
            # Fast loading with soundfile
            clean, sr_clean = sf.read(clean_path, dtype='float32')
            noisy, sr_noisy = sf.read(noisy_path, dtype='float32')
            
            # Validate sample rates
            if sr_clean != self.sample_rate or sr_noisy != self.sample_rate:
                error_msg = (
                    f"{filename}: Expected {self.sample_rate}Hz, "
                    f"got clean={sr_clean}Hz, noisy={sr_noisy}Hz"
                )
                
                if self.skip_on_error:
                    warnings.warn(f"Skipping {error_msg}")
                    self.error_count += 1
                    return self.__getitem__((idx + 1) % len(self))
                else:
                    raise ValueError(error_msg)
            
        except Exception as e:
            if self.skip_on_error:
                warnings.warn(f"Failed to load {filename}: {e}. Skipping.")
                self.error_count += 1
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise IOError(f"Failed to load {filename}: {e}")
        
        # Ensure same length
        min_len = min(len(clean), len(noisy))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        
        # Store original length
        original_length = len(clean)
        
        # Crop if too long (deterministic or random based on mode)
        if len(clean) > self.max_length_samples:
            if self.mode == 'train' and self.random_crop:
                # Random crop for data augmentation
                max_start = len(clean) - self.max_length_samples
                start = random.randint(0, max_start)
            else:
                # Deterministic crop from beginning (for eval/test)
                start = 0
            
            clean = clean[start:start + self.max_length_samples]
            noisy = noisy[start:start + self.max_length_samples]
        
        # NO NORMALIZATION - Best practice for speech enhancement!
        # This preserves the SNR relationships in the data
        
        sample = {
            'mix': torch.from_numpy(noisy).float(),
            'sph': torch.from_numpy(clean).float(),
            'n_samples': len(clean),
            'original_length': min(original_length, self.max_length_samples),
            'filename': filename
        }
        
        if self.memory_cache is not None:
            self.memory_cache[idx] = sample
        
        return sample


class SegmentDataset(Dataset):
    """
    Efficient segment extraction from utterances
    
    Segments are computed on-the-fly to save memory.
    For best performance with overlapping segments, enable
    cache_in_memory on the base dataset.
    
    ? FIXED: Now works correctly with cropped utterances
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        segment_size: float = 6.0,
        hop_size: float = 1.0,
        sample_rate: int = 16000,
        include_trailing: bool = True
    ):
        """
        Args:
            base_dataset: Underlying utterance-level dataset
            segment_size: Segment length in seconds
            hop_size: Hop between segments in seconds
            sample_rate: Sample rate
            include_trailing: Whether to include trailing partial segment
        """
        self.base_dataset = base_dataset
        self.seg_len = int(segment_size * sample_rate)
        self.hop_len = int(hop_size * sample_rate)
        self.include_trailing = include_trailing
        
        # Pre-compute segment indices (lightweight)
        self.segment_map = self._build_segment_map()
        
        # Warn about performance if many overlapping segments
        if len(self.segment_map) > len(base_dataset) * 10:
            warnings.warn(
                f"Created {len(self.segment_map)} segments from "
                f"{len(base_dataset)} utterances. Consider enabling "
                f"cache_in_memory on base_dataset for better performance."
            )
        
        print(f"Created {len(self.segment_map)} segments from {len(base_dataset)} utterances")
    
    def _build_segment_map(self) -> List[Tuple[int, int, int]]:
        """
        Build map of (utterance_idx, segment_start, segment_end)
        
        ? FIXED: Now uses capped lengths from base_dataset, preventing invalid segments
        """
        segment_map = []
        
        # Statistics for logging
        total_utts = len(self.base_dataset)
        empty_utts = 0
        short_utts = 0
        long_utts = 0
        
        for utt_idx in range(total_utts):
            # ? Use lengths from base_dataset (already capped at max_length_samples)
            n_samples = self.base_dataset.lengths[utt_idx]
            
            if n_samples == 0:
                warnings.warn(f"Utterance {utt_idx} has length 0, skipping")
                empty_utts += 1
                continue
            
            if n_samples < self.seg_len:
                # Short utterance - will be padded
                segment_map.append((utt_idx, 0, n_samples))
                short_utts += 1
            else:
                # Long utterance - split into segments
                start = 0
                utt_segments = 0
                
                while start < n_samples:
                    end = min(start + self.seg_len, n_samples)
                    seg_len = end - start
                    
                    # Add segment if full or trailing allowed
                    if seg_len == self.seg_len or (self.include_trailing and seg_len > 0):
                        segment_map.append((utt_idx, start, end))
                        utt_segments += 1
                    
                    if end >= n_samples:
                        break
                    
                    start += self.hop_len
                
                if utt_segments > 0:
                    long_utts += 1
        
        # Log statistics
        print(f"\n{'='*60}")
        print(f"SEGMENT MAP STATISTICS")
        print(f"{'='*60}")
        print(f"Total utterances: {total_utts}")
        print(f"  Short (< {self.seg_len} samples): {short_utts} ? 1 segment each (padded)")
        print(f"  Long  (= {self.seg_len} samples): {long_utts} ? multiple segments")
        print(f"  Empty (0 samples): {empty_utts} ? skipped")
        print(f"Total segments created: {len(segment_map)}")
        print(f"Avg segments per utterance: {len(segment_map) / max(total_utts - empty_utts, 1):.2f}")
        print(f"{'='*60}\n")
        
        return segment_map
    
    def __len__(self) -> int:
        return len(self.segment_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load one segment with padding if needed
        
        ? FIXED: Now correctly handles segments that won't exceed available data
        """
        try:
            utt_idx, seg_start, seg_end = self.segment_map[idx]
        except IndexError:
            raise IndexError(f"Segment index {idx} out of range")
        
        # Load full utterance
        try:
            sample = self.base_dataset[utt_idx]
        except Exception as e:
            raise IOError(f"Failed to load utterance {utt_idx} for segment {idx}: {e}")
        
        # Extract segment
        mix = sample['mix'][seg_start:seg_end]
        sph = sample['sph'][seg_start:seg_end]
        actual_len = len(mix)
        
        # Safety check (should never trigger after fix)
        if actual_len == 0:
            warnings.warn(
                f"Empty segment detected! utt={utt_idx}, seg_start={seg_start}, "
                f"seg_end={seg_end}, sample_len={len(sample['mix'])}"
            )
            # Return a minimal valid segment
            actual_len = 1
            mix = torch.zeros(self.seg_len)
            sph = torch.zeros(self.seg_len)
        
        # Pad if needed
        if len(mix) < self.seg_len:
            pad_size = self.seg_len - len(mix)
            mix = F.pad(mix, (0, pad_size), value=0)
            sph = F.pad(sph, (0, pad_size), value=0)
        
        return {
            'mix': mix,
            'sph': sph,
            'n_samples': actual_len,
            'filename': sample['filename']
        }


def collate_fn_segments(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for segment-based batching"""
    try:
        mix_batch = torch.stack([item['mix'] for item in batch])
        sph_batch = torch.stack([item['sph'] for item in batch])
        n_samples = torch.tensor([item['n_samples'] for item in batch], dtype=torch.int64)
        filenames = [item['filename'] for item in batch]
        
        return {
            'mix': mix_batch,
            'sph': sph_batch,
            'n_samples': n_samples,
            'filenames': filenames
        }
    except Exception as e:
        raise RuntimeError(f"Error in collate_fn_segments: {e}")


def collate_fn_utterances(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for utterance-based batching with padding"""
    try:
        # Find max length in batch
        max_len = max([item['mix'].shape[0] for item in batch])
        
        mix_list = []
        sph_list = []
        n_samples = []
        filenames = []
        
        for item in batch:
            mix = item['mix']
            sph = item['sph']
            
            # Pad to max length in batch
            if len(mix) < max_len:
                pad_size = max_len - len(mix)
                mix = F.pad(mix, (0, pad_size), value=0)
                sph = F.pad(sph, (0, pad_size), value=0)
            
            mix_list.append(mix)
            sph_list.append(sph)
            n_samples.append(item['n_samples'])
            filenames.append(item['filename'])
        
        mix_batch = torch.stack(mix_list)
        sph_batch = torch.stack(sph_list)
        
        return {
            'mix': mix_batch,
            'sph': sph_batch,
            'n_samples': torch.tensor(n_samples, dtype=torch.int64),
            'filenames': filenames
        }
    except Exception as e:
        raise RuntimeError(f"Error in collate_fn_utterances: {e}")


def create_dataloaders(
    train_clean_dir: str,
    train_noisy_dir: str,
    valid_clean_dir: str,
    valid_noisy_dir: str,
    test_clean_dir: str,
    test_noisy_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    sample_rate: int = 16000,
    unit: str = 'seg',
    segment_size: float = 6.0,
    segment_shift: float = 1.0,
    max_length_seconds: float = 6.0,
    pin_memory: bool = True,
    drop_last: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create optimal dataloaders for training from scratch (WITH PADDING FIX)
    
    BEST PRACTICES:
    1. NO normalization (preserves SNR)
    2. Fast soundfile loading (5-10x faster than librosa)
    3. Multi-worker support for parallel loading
    4. Pin memory for faster GPU transfer
    5. Smart caching for fast initialization
    6. Robust error handling
    7. ? FIXED: Proper length capping prevents invalid segments
    
    This configuration follows best practices from major papers:
    - SEGAN, Wave-U-Net, MetricGAN, DEMUCS, DCCRN
    
    Args:
        train_clean_dir: Training clean WAV directory
        train_noisy_dir: Training noisy WAV directory
        valid_clean_dir: Validation clean WAV directory
        valid_noisy_dir: Validation noisy WAV directory
        test_clean_dir: Test clean WAV directory
        test_noisy_dir: Test noisy WAV directory
        batch_size: Batch size
        num_workers: Number of parallel workers (4-8 recommended)
        sample_rate: Sample rate (16000 for VoiceBank+DEMAND)
        unit: 'seg' for segments, 'utt' for utterances
        segment_size: Segment size in seconds
        segment_shift: Hop size in seconds
        max_length_seconds: Maximum utterance length
        pin_memory: Use pinned memory for faster GPU transfer
        drop_last: Drop last incomplete batch in training
        cache_dir: Directory to cache file lengths
    
    Returns:
        train_loader, valid_loader, test_loader
    """
    
    assert unit in ['seg', 'utt'], f"Unit must be 'seg' or 'utt', got {unit}"
    
    # Setup cache files
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, 'train_lengths.pkl')
        valid_cache = os.path.join(cache_dir, 'valid_lengths.pkl')
        test_cache = os.path.join(cache_dir, 'test_lengths.pkl')
    else:
        train_cache = valid_cache = test_cache = None
    
    print("\n" + "="*60)
    print("INITIALIZING DATALOADERS - BEST PRACTICES (WITH PADDING FIX)")
    print("="*60)
    print("? NO normalization (preserves SNR)")
    print("? Fast soundfile loading")
    print("? Multi-worker parallel loading")
    print("? Pin memory for GPU speed")
    print("? Length capping prevents invalid segments")
    print("="*60)
    
    # Create base datasets
    train_base = SpeechEnhancementDataset(
        clean_dir=train_clean_dir,
        noisy_dir=train_noisy_dir,
        sample_rate=sample_rate,
        max_length_seconds=max_length_seconds,
        mode='train',
        random_crop=True,
        cache_in_memory=False,  # Don't cache large training set
        length_cache_file=train_cache,
        skip_on_error=True
    )
    
    valid_base = SpeechEnhancementDataset(
        clean_dir=valid_clean_dir,
        noisy_dir=valid_noisy_dir,
        sample_rate=sample_rate,
        max_length_seconds=max_length_seconds,
        mode='eval',
        random_crop=False,
        cache_in_memory=True,  # Cache small validation set
        length_cache_file=valid_cache,
        skip_on_error=True
    )
    
    test_base = SpeechEnhancementDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        max_length_seconds=max_length_seconds,
        mode='eval',
        random_crop=False,
        cache_in_memory=True,  # Cache small test set
        length_cache_file=test_cache,
        skip_on_error=False  # Strict for testing
    )
    
    # Wrap in SegmentDataset if needed
    if unit == 'seg':
        print("\n" + "="*60)
        print("CREATING SEGMENT DATASETS")
        print("="*60)
        
        train_dataset = SegmentDataset(
            train_base,
            segment_size=segment_size,
            hop_size=segment_shift,
            sample_rate=sample_rate,
            include_trailing=True
        )
        
        valid_dataset = SegmentDataset(
            valid_base,
            segment_size=segment_size,
            hop_size=segment_shift,
            sample_rate=sample_rate,
            include_trailing=True
        )
        
        test_dataset = SegmentDataset(
            test_base,
            segment_size=segment_size,
            hop_size=segment_shift,
            sample_rate=sample_rate,
            include_trailing=True
        )
        
        collate_fn = collate_fn_segments
    else:
        train_dataset = train_base
        valid_dataset = valid_base
        test_dataset = test_base
        collate_fn = collate_fn_utterances
    
    # Check persistent_workers support
    use_persistent = PERSISTENT_WORKERS_SUPPORTED and num_workers > 0
    
    if num_workers > 0 and not PERSISTENT_WORKERS_SUPPORTED:
        warnings.warn(
            "PyTorch < 1.7 does not support persistent_workers. "
            "Update PyTorch for better performance."
        )
    
    # Create DataLoaders with optimal settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=use_persistent
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1 if unit == 'utt' else batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1 if unit == 'utt' else batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent
    )
    
    print("\n" + "="*60)
    print("? DATALOADERS CREATED - READY FOR TRAINING!")
    print("="*60)
    print(f"Configuration:")
    print(f"  Unit: {unit}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Pin memory: {pin_memory}")
    print(f"  Persistent workers: {use_persistent}")
    print(f"  Normalization: None (best practice)")
    if unit == 'seg':
        print(f"  Segment size: {segment_size}s ({int(segment_size * sample_rate)} samples)")
        print(f"  Segment shift: {segment_shift}s ({int(segment_shift * sample_rate)} samples)")
    print(f"  Max length: {max_length_seconds}s")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_base)} utterances ? {len(train_dataset)} samples ? {len(train_loader)} batches")
    print(f"  Valid: {len(valid_base)} utterances ? {len(valid_dataset)} samples ? {len(valid_loader)} batches")
    print(f"  Test:  {len(test_base)} utterances ? {len(test_dataset)} samples ? {len(test_loader)} batches")
    print("="*60 + "\n")
    
    return train_loader, valid_loader, test_loader


def create_test_dataloader_only(
    test_clean_dir: str,
    test_noisy_dir: str,
    batch_size: int = 1,
    num_workers: int = 2,
    sample_rate: int = 16000,
    unit: str = 'seg',
    segment_size: float = 6.0,
    segment_shift: float = 6.0,
    max_length_seconds: float = 6.0,
    pin_memory: bool = True,
    cache_dir: Optional[str] = None
) -> DataLoader:
    """
    Create test-only dataloader (efficient for testing) WITH PADDING FIX
    
    Uses same best practices as training loaders:
    - NO normalization
    - Fast loading
    - Smart caching
    - ? Length capping prevents invalid segments
    """
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        test_cache = os.path.join(cache_dir, 'test_lengths.pkl')
    else:
        test_cache = None
    
    print("\n" + "="*60)
    print("CREATING TEST DATALOADER - BEST PRACTICES (WITH PADDING FIX)")
    print("="*60)
    
    test_base = SpeechEnhancementDataset(
        clean_dir=test_clean_dir,
        noisy_dir=test_noisy_dir,
        sample_rate=sample_rate,
        max_length_seconds=max_length_seconds,
        mode='eval',
        random_crop=False,
        cache_in_memory=True,
        length_cache_file=test_cache,
        skip_on_error=False
    )
    
    if unit == 'seg':
        test_dataset = SegmentDataset(
            test_base,
            segment_size=segment_size,
            hop_size=segment_shift,
            sample_rate=sample_rate,
            include_trailing=True
        )
        collate_fn = collate_fn_segments
    else:
        test_dataset = test_base
        collate_fn = collate_fn_utterances
    
    use_persistent = PERSISTENT_WORKERS_SUPPORTED and num_workers > 0
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent
    )
    
    print(f"Test: {len(test_base)} utterances ? {len(test_dataset)} samples ? {len(test_loader)} batches")
    print(f"Normalization: None (best practice)")
    print("="*60 + "\n")
    
    return test_loader


# ==================== USAGE EXAMPLE ====================

if __name__ == '__main__':
    """
    Example usage for VoiceBank+DEMAND
    """
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_clean_dir='/path/to/VoiceBank/train/clean',
        train_noisy_dir='/path/to/VoiceBank/train/noisy',
        valid_clean_dir='/path/to/VoiceBank/valid/clean',
        valid_noisy_dir='/path/to/VoiceBank/valid/noisy',
        test_clean_dir='/path/to/VoiceBank/test/clean',
        test_noisy_dir='/path/to/VoiceBank/test/noisy',
        batch_size=4,
        num_workers=4,
        sample_rate=16000,
        unit='seg',
        segment_size=6.0,
        segment_shift=1.0,
        max_length_seconds=6.0,
        pin_memory=True,
        cache_dir='./cache'
    )
    
    # Training loop example
    print("\nTesting data loading...")
    for batch in train_loader:
        mix = batch['mix']  # [batch_size, time]
        sph = batch['sph']  # [batch_size, time]
        n_samples = batch['n_samples']  # [batch_size]
        
        print(f"Batch shape: mix={mix.shape}, sph={sph.shape}")
        print(f"Sample lengths: {n_samples}")
        print(f"Value ranges: mix=[{mix.min():.3f}, {mix.max():.3f}], "
              f"sph=[{sph.min():.3f}, {sph.max():.3f}]")
        print("? Data loading successful!\n")
        break