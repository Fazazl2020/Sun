"""
configs_ab2.py - Ablation 2: InstanceNorm Fix + Residual Connection
====================================================================
BEFORE RUNNING:
  cp utils/networks_ab2.py utils/networks.py

This ablation tests InstanceNorm fix + residual connection.
Network: networks_ab2.py (with residual: output = input + decoded)
"""

import os
from pathlib import Path

# ============================================================
# ==================== ABLATION CONFIGURATION ================
# ============================================================

ABLATION_NAME = 'Ab2-Residual'
ABLATION_DESC = 'InstanceNorm fix + Residual connection (output = input + decoded)'

# Using S2 config (F=201) - best performing
EXPERIMENT = 'S2'

EXPERIMENTS = {
    'S2': {
        'name': 'Stage1-STFT201',
        'description': 'F=201 (higher resolution)',
        'stft_win': 0.025,      # 400 samples -> F=201
        'stft_hop': 0.00625,    # 100 samples
        'use_compressed_mag': False,
        'mag_compression_power': 0.3,
    },
}

EXP_CONFIG = EXPERIMENTS[EXPERIMENT]

# ============================================================
# ==================== PATHS =================================
# ============================================================

DATASET_ROOT = '/gdata/fewahab/data/Voicebank+demand/My_train_valid_test'

TRAIN_CLEAN_SUBDIR = 'Train/clean_train'
TRAIN_NOISY_SUBDIR = 'Train/noisy_train'
VALID_CLEAN_SUBDIR = 'valid/clean_valid'
VALID_NOISY_SUBDIR = 'valid/noisy_valid'
TEST_CLEAN_SUBDIR = 'Test/clean_test'
TEST_NOISY_SUBDIR = 'Test/noisy_test'

# ABLATION-SPECIFIC CHECKPOINT PATH
CHECKPOINT_ROOT = '/ghome/fewahab/Sun-Models/Mod-3/T71a1/scripts/ckpt_ab2'

ESTIMATES_ROOT = '/gdata/fewahab/Sun-Models/Mod-3/T71a1/estimates_ab2'

# ============================================================
# ==================== MODEL CONFIG ==========================
# ============================================================

MODEL_CONFIG = {
    'in_norm': False,
    'sample_rate': 16000,
    'win_len': EXP_CONFIG['stft_win'],
    'hop_len': EXP_CONFIG['stft_hop'],
    'use_compressed_mag': EXP_CONFIG['use_compressed_mag'],
    'mag_compression_power': EXP_CONFIG['mag_compression_power'],
}

# ============================================================
# ==================== TRAINING CONFIG =======================
# ============================================================

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
    'max_n_epochs': 200,
    'early_stop_patience': 50,
    'clip_norm': 1.0,
    'loss_log': 'loss.txt',
    'time_log': '',
    'resume_model': '',
    'pesq_eval_interval': 10,
    'pesq_log': 'pesq_log.txt',
    'save_best_pesq_model': True,
}

TESTING_CONFIG = {
    'batch_size': 1,
    'num_workers': 2,
    'write_ideal': False,
}

# ============================================================
# ==================== DERIVED PATHS =========================
# ============================================================

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
    'pesq_eval_interval': TRAINING_CONFIG['pesq_eval_interval'],
    'pesq_log': TRAINING_CONFIG['pesq_log'],
    'save_best_pesq_model': TRAINING_CONFIG['save_best_pesq_model'],
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
    print(f"\n{'='*60}")
    print(f"ABLATION: {ABLATION_NAME}")
    print(f"{ABLATION_DESC}")
    print(f"{'='*60}")

    if mode in ['train', 'all']:
        validate_path(TRAIN_CLEAN_DIR, "Training clean directory")
        validate_path(TRAIN_NOISY_DIR, "Training noisy directory")

    if mode in ['valid', 'train', 'all']:
        validate_path(VALID_CLEAN_DIR, "Validation clean directory")
        validate_path(VALID_NOISY_DIR, "Validation noisy directory")

    if mode in ['test', 'all']:
        validate_path(TEST_CLEAN_DIR, "Test clean directory")
        validate_path(TEST_NOISY_DIR, "Test noisy directory")

    print(f"Data directories validated!")
    print(f"{'='*60}\n")

print(f"\n{'='*60}")
print(f"ABLATION: {ABLATION_NAME}")
print(f"{ABLATION_DESC}")
print(f"Checkpoint: {CHECKPOINT_ROOT}")
print(f"{'='*60}\n")
