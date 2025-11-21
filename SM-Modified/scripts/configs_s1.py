"""
configs_s1.py - Experiment S1: Stage1 Baseline
==============================================
F=161, Standard Magnitude Loss

USAGE: cp configs_s1.py configs.py && python train.py
"""

import os

EXPERIMENT_NAME = 'S1-Baseline'

# ============================================================
# ==================== EASY-TO-MODIFY SETTINGS ===============
# ============================================================

# PESQ Validation Settings (modify these as needed)
PESQ_EVAL_INTERVAL = 10       # Evaluate PESQ every N epochs (0 = disabled)
PESQ_LOG = 'pesq_log.txt'     # PESQ results log file
SAVE_BEST_PESQ_MODEL = True   # Save separate best_pesq.pt model

# Dataset
DATASET_ROOT = '/gdata/fewahab/data/Voicebank+demand/My_train_valid_test'
TRAIN_CLEAN_SUBDIR = 'train/clean'
TRAIN_NOISY_SUBDIR = 'train/noisy'
VALID_CLEAN_SUBDIR = 'valid/clean'
VALID_NOISY_SUBDIR = 'valid/noisy'
TEST_CLEAN_SUBDIR = 'test/clean'
TEST_NOISY_SUBDIR = 'test/noisy'

# Checkpoints - UPDATE THIS FOR YOUR EXPERIMENT
CHECKPOINT_ROOT = '/ghome/fewahab/My_5th_pap/Ab3/N13/scripts/ckpt'
ESTIMATES_ROOT = '/gdata/fewahab/My_5th_pap/Ab3/N13/scripts/estimates'

# ============================================================
# ==================== MODEL CONFIGURATION ===================
# ============================================================

# Model Config - S1: F=161, standard mag loss
MODEL_CONFIG = {
    'in_norm': False,
    'sample_rate': 16000,
    'win_len': 0.020,       # 320 samples -> F=161
    'hop_len': 0.010,       # 160 samples
    'use_compressed_mag': False,
    'mag_compression_power': 0.3,
}

# Training Config
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
    # PESQ Validation Settings (from top of file)
    'pesq_eval_interval': PESQ_EVAL_INTERVAL,
    'pesq_log': PESQ_LOG,
    'save_best_pesq_model': SAVE_BEST_PESQ_MODEL,
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
    # PESQ validation settings
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
        raise FileNotFoundError(f"{path_type} not found: {path}")
    return path

def validate_data_dirs(mode='train'):
    if mode in ['train', 'all']:
        validate_path(TRAIN_CLEAN_DIR)
        validate_path(TRAIN_NOISY_DIR)
    if mode in ['valid', 'train', 'all']:
        validate_path(VALID_CLEAN_DIR)
        validate_path(VALID_NOISY_DIR)
    if mode in ['test', 'all']:
        validate_path(TEST_CLEAN_DIR)
        validate_path(TEST_NOISY_DIR)

def check_pytorch_version():
    import torch
    return {'version': torch.__version__, 'cuda_available': torch.cuda.is_available()}

def print_config():
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"STFT: F=161 (win=320, hop=160)")
    print(f"Mag Loss: Standard")
    print(f"PESQ Eval: Every {PESQ_EVAL_INTERVAL} epochs")
    print(f"Checkpoint: {CHECKPOINT_ROOT}")
    print(f"{'='*60}\n")

print_config()
