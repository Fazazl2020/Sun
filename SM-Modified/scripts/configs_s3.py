"""
configs_s3.py - Experiment S3: Compressed Magnitude Loss
========================================================
InstanceNorm + Compressed MagLoss (mag^0.3), F=161

Loss: SI-SDR + 10.0 * L1(R,I) + 3.0 * MSE(Magnitude^0.3)

CHANGES FROM S1:
- Magnitude loss computed on compressed magnitudes (mag^0.3)
- This weights small magnitude errors more equally with large ones
- Perceptually motivated: quiet sounds matter too

RATIONALE:
Raw MSE on magnitude:
  - mag=100 vs 101 -> error=1
  - mag=0.1 vs 0.2 -> error=0.01 (100% relative error but tiny MSE!)

Compressed MSE (mag^0.3):
  - 100^0.3=4.0, 101^0.3=4.03 -> error=0.03
  - 0.1^0.3=0.46, 0.2^0.3=0.62 -> error=0.16
  - Now small magnitudes get proper attention!

NOTE: This is different from CMGAN's full pipeline compression.
CMGAN compresses input -> model -> decompresses output.
We only compress for the loss computation.

USAGE:
1. Copy this file to configs.py: cp configs_s3.py configs.py
2. Update CHECKPOINT_ROOT to your experiment directory
3. Run: python train.py
"""

import os
from pathlib import Path

# ============================================================
# ==================== EXPERIMENT INFO =======================
# ============================================================

EXPERIMENT_NAME = 'S3-CompressedMagLoss'
EXPERIMENT_DESC = 'InstanceNorm + Compressed MagLoss (mag^0.3), F=161'

# ============================================================
# ==================== USER CONFIGURATION ====================
# ============================================================

# ==================== 1. DATASET PATHS ====================

DATASET_ROOT = '/gdata/fewahab/data/Voicebank+demand/My_train_valid_test'

# FIXED SUBDIRECTORIES
TRAIN_CLEAN_SUBDIR = 'Train/clean_train'
TRAIN_NOISY_SUBDIR = 'Train/noisy_train'
VALID_CLEAN_SUBDIR = 'valid/clean_valid'
VALID_NOISY_SUBDIR = 'valid/noisy_valid'
TEST_CLEAN_SUBDIR = 'Test/clean_test'
TEST_NOISY_SUBDIR = 'Test/noisy_test'


# ==================== 2. CHECKPOINT PATHS ====================
# IMPORTANT: Change this for each experiment to avoid file collision!

CHECKPOINT_ROOT = '/ghome/fewahab/Sun-Models/Exp-S3/ckpt'


# ==================== 3. OUTPUT PATHS ====================

ESTIMATES_ROOT = '/gdata/fewahab/Sun-Models/Exp-S3/estimates'


# ==================== 4. MODEL CONFIGURATION ====================

MODEL_CONFIG = {
    'in_norm': False,
    'sample_rate': 16000,
    # S3: Standard STFT (F=161)
    'win_len': 0.020,       # 320 samples -> F=161
    'hop_len': 0.010,       # 160 samples
    # S3: Compressed magnitude loss (mag^0.3)
    'use_compressed_mag': True,
    'mag_compression_power': 0.3,
}


# ==================== 5. TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    'gpu_ids': '0',
    'unit': 'utt',
    'batch_size': 10,
    'num_workers': 4,
    'segment_size': 4.0,
    'segment_shift': 1.0,
    'max_length_seconds': 6.0,
    'lr': 0.001,
    'plateau_factor': 0.5,
    'plateau_patience': 15,
    'plateau_threshold': 0.001,
    'plateau_min_lr': 1e-6,
    'max_n_epochs': 800,
    'early_stop_patience': 50,
    'clip_norm': 1.0,
    'loss_log': 'loss.txt',
    'time_log': '',
    'resume_model': '',
}


# ==================== 6. TESTING CONFIGURATION ====================

TESTING_CONFIG = {
    'batch_size': 1,
    'num_workers': 2,
    'write_ideal': False,
}


# ============================================================
# ============ END OF USER CONFIGURATION =====================
# ============================================================

# Derived paths
TRAIN_CLEAN_DIR = os.path.join(DATASET_ROOT, TRAIN_CLEAN_SUBDIR)
TRAIN_NOISY_DIR = os.path.join(DATASET_ROOT, TRAIN_NOISY_SUBDIR)
VALID_CLEAN_DIR = os.path.join(DATASET_ROOT, VALID_CLEAN_SUBDIR)
VALID_NOISY_DIR = os.path.join(DATASET_ROOT, VALID_NOISY_SUBDIR)
TEST_CLEAN_DIR = os.path.join(DATASET_ROOT, TEST_CLEAN_SUBDIR)
TEST_NOISY_DIR = os.path.join(DATASET_ROOT, TEST_NOISY_SUBDIR)

CHECKPOINT_DIR = CHECKPOINT_ROOT
LOGS_DIR = os.path.join(CHECKPOINT_DIR, 'logs')
MODELS_DIR = os.path.join(CHECKPOINT_DIR, 'models')
CACHE_DIR = os.path.join(CHECKPOINT_DIR, 'cache')
ESTIMATES_DIR = ESTIMATES_ROOT

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(ESTIMATES_DIR, exist_ok=True)

exp_conf = MODEL_CONFIG

train_conf = {
    'gpu_ids': TRAINING_CONFIG['gpu_ids'],
    'ckpt_dir': CHECKPOINT_DIR,
    'est_path': ESTIMATES_DIR,
    'unit': TRAINING_CONFIG['unit'],
    'batch_size': TRAINING_CONFIG['batch_size'],
    'num_workers': TRAINING_CONFIG['num_workers'],
    'segment_size': TRAINING_CONFIG['segment_size'],
    'segment_shift': TRAINING_CONFIG['segment_shift'],
    'max_length_seconds': TRAINING_CONFIG['max_length_seconds'],
    'lr': TRAINING_CONFIG['lr'],
    'plateau_factor': TRAINING_CONFIG['plateau_factor'],
    'plateau_patience': TRAINING_CONFIG['plateau_patience'],
    'plateau_threshold': TRAINING_CONFIG['plateau_threshold'],
    'plateau_min_lr': TRAINING_CONFIG['plateau_min_lr'],
    'max_n_epochs': TRAINING_CONFIG['max_n_epochs'],
    'early_stop_patience': TRAINING_CONFIG['early_stop_patience'],
    'clip_norm': TRAINING_CONFIG['clip_norm'],
    'loss_log': TRAINING_CONFIG['loss_log'],
    'time_log': TRAINING_CONFIG['time_log'],
    'resume_model': TRAINING_CONFIG['resume_model'],
}

test_conf = {
    'model_file': os.path.join(MODELS_DIR, 'best.pt'),
    'batch_size': TESTING_CONFIG['batch_size'],
    'num_workers': TESTING_CONFIG['num_workers'],
    'write_ideal': TESTING_CONFIG['write_ideal'],
}

def validate_path(path, path_type="directory"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path_type.capitalize()} not found: {path}")
    return path

def validate_data_dirs(mode='train'):
    print("\n" + "="*60)
    print("VALIDATING DATA DIRECTORIES")
    print("="*60)

    if mode in ['train', 'all']:
        print(f"\nChecking training data...")
        print(f"  Clean: {TRAIN_CLEAN_DIR}")
        print(f"  Noisy: {TRAIN_NOISY_DIR}")

        validate_path(TRAIN_CLEAN_DIR, "Training clean directory")
        validate_path(TRAIN_NOISY_DIR, "Training noisy directory")

        train_clean_files = [f for f in os.listdir(TRAIN_CLEAN_DIR) if f.endswith('.wav')]
        train_noisy_files = [f for f in os.listdir(TRAIN_NOISY_DIR) if f.endswith('.wav')]

        if len(train_clean_files) == 0:
            raise ValueError(f"No WAV files found in {TRAIN_CLEAN_DIR}")
        if len(train_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {TRAIN_NOISY_DIR}")

        print(f"  Found {len(train_clean_files)} clean files")
        print(f"  Found {len(train_noisy_files)} noisy files")

    if mode in ['valid', 'train', 'all']:
        print(f"\nChecking validation data...")
        print(f"  Clean: {VALID_CLEAN_DIR}")
        print(f"  Noisy: {VALID_NOISY_DIR}")

        validate_path(VALID_CLEAN_DIR, "Validation clean directory")
        validate_path(VALID_NOISY_DIR, "Validation noisy directory")

        valid_clean_files = [f for f in os.listdir(VALID_CLEAN_DIR) if f.endswith('.wav')]
        valid_noisy_files = [f for f in os.listdir(VALID_NOISY_DIR) if f.endswith('.wav')]

        if len(valid_clean_files) == 0:
            raise ValueError(f"No WAV files found in {VALID_CLEAN_DIR}")
        if len(valid_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {VALID_NOISY_DIR}")

        print(f"  Found {len(valid_clean_files)} clean files")
        print(f"  Found {len(valid_noisy_files)} noisy files")

    if mode in ['test', 'all']:
        print(f"\nChecking test data...")
        print(f"  Clean: {TEST_CLEAN_DIR}")
        print(f"  Noisy: {TEST_NOISY_DIR}")

        validate_path(TEST_CLEAN_DIR, "Test clean directory")
        validate_path(TEST_NOISY_DIR, "Test noisy directory")

        test_clean_files = [f for f in os.listdir(TEST_CLEAN_DIR) if f.endswith('.wav')]
        test_noisy_files = [f for f in os.listdir(TEST_NOISY_DIR) if f.endswith('.wav')]

        if len(test_clean_files) == 0:
            raise ValueError(f"No WAV files found in {TEST_CLEAN_DIR}")
        if len(test_noisy_files) == 0:
            raise ValueError(f"No WAV files found in {TEST_NOISY_DIR}")

        print(f"  Found {len(test_clean_files)} clean files")
        print(f"  Found {len(test_noisy_files)} noisy files")

    print("\n" + "="*60)
    print("ALL DATA DIRECTORIES VALIDATED SUCCESSFULLY!")
    print("="*60 + "\n")

def check_pytorch_version():
    try:
        import torch
        version = torch.__version__.split('+')[0]
        major, minor, patch = map(int, version.split('.'))
        persistent_workers_supported = (major > 1) or (major == 1 and minor >= 7)
        return {
            'version': torch.__version__,
            'persistent_workers': persistent_workers_supported,
            'cuda_available': torch.cuda.is_available()
        }
    except ImportError:
        raise ImportError("PyTorch is not installed!")

def print_config():
    pytorch_info = check_pytorch_version()

    # Compute F for display
    sample_rate = MODEL_CONFIG['sample_rate']
    win_size = int(MODEL_CONFIG['win_len'] * sample_rate)
    hop_size = int(MODEL_CONFIG['hop_len'] * sample_rate)
    F = win_size // 2 + 1

    print("\n" + "="*70)
    print("CONFIGURATION LOADED")
    print("="*70)
    print(f"\n** EXPERIMENT: {EXPERIMENT_NAME} **")
    print(f"   {EXPERIMENT_DESC}")
    print(f"\n   STFT: win={win_size}, hop={hop_size}, F={F}")
    print(f"   Compressed Mag Loss: {MODEL_CONFIG['use_compressed_mag']} (power={MODEL_CONFIG['mag_compression_power']})")
    print(f"\nPYTORCH INFO:")
    print(f"  Version: {pytorch_info['version']}")
    print(f"  CUDA available: {pytorch_info['cuda_available']}")
    print(f"\nDIRECTORIES:")
    print(f"  Dataset:     {DATASET_ROOT}")
    print(f"  Checkpoints: {CHECKPOINT_ROOT}")
    print(f"  Outputs:     {ESTIMATES_ROOT}")
    print(f"\nTRAINING CONFIG:")
    print(f"  GPU: {TRAINING_CONFIG['gpu_ids']}")
    print(f"  Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  Max length: {TRAINING_CONFIG['max_length_seconds']}s")
    print(f"  Max epochs: {TRAINING_CONFIG['max_n_epochs']}")
    print("="*70 + "\n")

print_config()
