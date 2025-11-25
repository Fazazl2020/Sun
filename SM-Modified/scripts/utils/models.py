"""
model.py - PRODUCTION-READY VERSION WITH PADDING FIX + PESQ VALIDATION
Perfect integration with data_utils.py - All bugs fixed
Includes PESQ validation on full validation set every N epochs
"""

import os
import shutil
import timeit
import numpy as np
import soundfile as sf
import torch
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

# PESQ for perceptual quality evaluation
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("WARNING: pesq not installed. Run 'pip install pesq' to enable PESQ validation.")

from configs import (
    exp_conf, train_conf, test_conf,
    TRAIN_CLEAN_DIR, TRAIN_NOISY_DIR,
    VALID_CLEAN_DIR, VALID_NOISY_DIR,
    TEST_CLEAN_DIR, TEST_NOISY_DIR,
    validate_data_dirs
)
from utils.utils import getLogger, numParams, countFrames, lossMask, wavNormalize
from utils.pipeline_modules import NetFeeder, Resynthesizer
from utils.data_utils import create_dataloaders, create_test_dataloader_only
from utils.networks import Net
from utils.criteria import LossFunction


class CheckPoint(object):
    """Checkpoint management with scheduler state support"""
    
    def __init__(self, ckpt_info=None, net_state_dict=None, 
                 optim_state_dict=None, scheduler_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict
        self.scheduler_state_dict = scheduler_state_dict
    
    def save(self, filename, is_best, best_model=None):
        """Save checkpoint to file"""
        torch.save(self, filename)
        if is_best and best_model:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        """Load checkpoint from file"""
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'No checkpoint found at {filename}')
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info 
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict
        self.scheduler_state_dict = getattr(ckpt, 'scheduler_state_dict', None)


def lossLog(log_file, ckpt, logging_period):
    """Write loss to CSV file"""
    ckpt_info = ckpt.ckpt_info

    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch, iter, tr_loss, cv_loss\n')

    with open(log_file, 'a') as f:
        f.write('{}, {}, {:.4f}, {:.4f}\n'.format(
            ckpt_info['cur_epoch'] + 1,
            ckpt_info['cur_iter'] + 1,
            ckpt_info['tr_loss'],
            ckpt_info['cv_loss']
        ))


def pesqLog(log_file, epoch, avg_pesq, best_pesq, n_samples, eval_time):
    """Write PESQ validation results to log file"""
    if not os.path.isfile(log_file):
        with open(log_file, 'w') as f:
            f.write('epoch, avg_pesq, best_pesq, n_samples, eval_time_sec\n')

    with open(log_file, 'a') as f:
        f.write('{}, {:.4f}, {:.4f}, {}, {:.1f}\n'.format(
            epoch, avg_pesq, best_pesq, n_samples, eval_time
        ))


class Model(object):
    """
    Main model class - PRODUCTION-READY WITH PADDING FIX
    Uses direct folders with optimal data loading
    """
    
    def __init__(self):
        """Initialize model with configuration"""
        # Get configuration from config file
        self.in_norm = exp_conf['in_norm']
        self.sample_rate = exp_conf['sample_rate']
        self.win_len = exp_conf['win_len']
        self.hop_len = exp_conf['hop_len']

        self.win_size = int(self.win_len * self.sample_rate)
        self.hop_size = int(self.hop_len * self.sample_rate)
        self.F = self.win_size // 2 + 1  # Number of frequency bins
    
    def train(self):
        """
        Training procedure with optimal data loading
        """
        # Validate data directories first
        print("\n" + "="*70)
        print("STEP 1: VALIDATING DATA DIRECTORIES")
        print("="*70)
        try:
            validate_data_dirs(mode='train')
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? DATA VALIDATION FAILED: {e}")
            print("\nPlease check your configs.py paths!")
            return
        
        # Load configuration
        self.ckpt_dir = train_conf['ckpt_dir']
        self.resume_model = train_conf['resume_model']
        self.time_log = train_conf['time_log']
        self.lr = train_conf['lr']
        self.plateau_factor = train_conf['plateau_factor']
        self.plateau_patience = train_conf['plateau_patience']
        self.plateau_threshold = train_conf['plateau_threshold']
        self.plateau_min_lr = train_conf['plateau_min_lr']
        self.clip_norm = train_conf['clip_norm']
        self.max_n_epochs = train_conf['max_n_epochs']
        self.early_stop_patience = train_conf['early_stop_patience']
        self.batch_size = train_conf['batch_size']
        self.num_workers = train_conf['num_workers']
        self.loss_log = train_conf['loss_log']
        self.unit = train_conf['unit']
        self.segment_size = train_conf['segment_size']
        self.segment_shift = train_conf['segment_shift']
        self.max_length_seconds = train_conf['max_length_seconds']

        # PESQ validation settings
        self.pesq_eval_interval = train_conf.get('pesq_eval_interval', 10)
        self.pesq_log = train_conf.get('pesq_log', 'pesq_log.txt')
        self.save_best_pesq_model = train_conf.get('save_best_pesq_model', True)

        # Setup device
        self.gpu_ids = tuple(map(int, train_conf['gpu_ids'].split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        # Create checkpoint directory
        os.makedirs(self.ckpt_dir, exist_ok=True)

        print("\n" + "="*70)
        print("STEP 2: INITIALIZING LOGGER")
        print("="*70)
        
        # Setup logger
        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)
        logger.info('='*70)
        logger.info('TRAINING CONFIGURATION')
        logger.info('='*70)
        logger.info(f'Training clean: {TRAIN_CLEAN_DIR}')
        logger.info(f'Training noisy: {TRAIN_NOISY_DIR}')
        logger.info(f'Validation clean: {VALID_CLEAN_DIR}')
        logger.info(f'Validation noisy: {VALID_NOISY_DIR}')
        logger.info(f'Sample rate: {self.sample_rate} Hz')
        logger.info(f'Normalization: {self.in_norm}')
        logger.info(f'Unit: {self.unit}')
        logger.info(f'Batch size: {self.batch_size}')
        logger.info(f'Num workers: {self.num_workers}')
        if self.unit == 'seg':
            logger.info(f'Segment size: {self.segment_size}s')
            logger.info(f'Segment shift: {self.segment_shift}s')
        logger.info(f'Max length: {self.max_length_seconds}s')
        logger.info(f'Initial LR: {self.lr}')
        logger.info(f'Max epochs: {self.max_n_epochs}')
        logger.info(f'Device: {self.device}')
        logger.info('='*70 + '\n')
        
        print("\n" + "="*70)
        print("STEP 3: CREATING DATALOADERS")
        print("="*70)
        
        # Setup cache directory for fast initialization
        cache_dir = os.path.join(self.ckpt_dir, 'cache')
        
        try:
            train_loader, valid_loader, _ = create_dataloaders(
                train_clean_dir=TRAIN_CLEAN_DIR,
                train_noisy_dir=TRAIN_NOISY_DIR,
                valid_clean_dir=VALID_CLEAN_DIR,
                valid_noisy_dir=VALID_NOISY_DIR,
                test_clean_dir=TEST_CLEAN_DIR,
                test_noisy_dir=TEST_NOISY_DIR,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sample_rate=self.sample_rate,
                unit=self.unit,
                segment_size=self.segment_size,
                segment_shift=self.segment_shift,
                max_length_seconds=self.max_length_seconds,
                pin_memory=True,
                drop_last=True,
                cache_dir=cache_dir
            )
        except Exception as e:
            logger.error(f'Failed to create dataloaders: {e}')
            print(f"\n? DATALOADER CREATION FAILED: {e}")
            raise
        
        # Calculate iterations per epoch
        self.logging_period = len(train_loader)
        logger.info(f'Iterations per epoch: {self.logging_period}\n')

        print("\n" + "="*70)
        print("STEP 4: INITIALIZING MODEL")
        print("="*70)

        # Create network with correct frequency bins
        logger.info(f'STFT config: win_size={self.win_size}, hop_size={self.hop_size}, F={self.F}')
        net = Net(F=self.F)
        logger.info(f'Model summary:\n{net}')

        net = net.to(self.device)
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)
            logger.info(f'Using DataParallel with {len(self.gpu_ids)} GPUs')

        # Calculate model size
        param_count = numParams(net)
        logger.info(f'Trainable parameters: {param_count:,d} -> {param_count*32/8/(2**20):.2f} MB\n')

        # Network feeder and resynthesizer (for PESQ validation)
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)

        # Loss and optimizer
        criterion = LossFunction(
            device=self.device, 
            win_size=self.win_size, 
            hop_size=self.hop_size
        )
        
        optimizer = Adam(net.parameters(), lr=self.lr, amsgrad=False)
        
        # ReduceLROnPlateau scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.plateau_factor,
            patience=self.plateau_patience,
            threshold=self.plateau_threshold,
            threshold_mode='rel',
            cooldown=2,
            min_lr=self.plateau_min_lr,
            verbose=True
        )
        
        # Initialize checkpoint info
        ckpt_info = {
            'cur_epoch': 0,
            'cur_iter': 0,
            'tr_loss': None,
            'cv_loss': None,
            'best_loss': float('inf'),
            'best_pesq': -1.0,  # Track best PESQ score
            'global_step': 0,
            'min_lr_epoch_count': 0
        }
        global_step = 0
        min_lr_epoch_count = 0
        best_pesq = -1.0
        
        # Resume training if needed
        if self.resume_model:
            logger.info('='*70)
            logger.info('RESUMING FROM CHECKPOINT')
            logger.info('='*70)
            logger.info(f'Loading: {self.resume_model}')
            
            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)
            
            # Load network state
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.' + key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)
            
            # Load optimizer state
            optim_state = ckpt.optim_state_dict
            for state in optim_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            optimizer.load_state_dict(optim_state)
            
            # Load checkpoint info
            ckpt_info = ckpt.ckpt_info
            global_step = ckpt_info.get('global_step', 0)
            min_lr_epoch_count = ckpt_info.get('min_lr_epoch_count', 0)
            best_pesq = ckpt_info.get('best_pesq', -1.0)

            logger.info(f'Resumed from epoch {ckpt_info["cur_epoch"] + 1}')
            logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
            logger.info(f'Best PESQ: {best_pesq:.4f}')
            logger.info('='*70 + '\n')
        
        print("\n" + "="*70)
        print("STEP 5: STARTING TRAINING LOOP")
        print("="*70 + "\n")
        
        # Training loop
        logger.info('Starting training loop...\n')
        
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0
            net.train()
            
            epoch_start_time = timeit.default_timer()
            
            # Iterate over batches
            for n_iter, batch in enumerate(train_loader):
                global_step += 1
                
                # Get batch data
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                
                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)

                iter_start_time = timeit.default_timer()

                # Prepare features and labels
                feat, lbl = feeder(mix, sph)
                loss_mask = lossMask(
                    shape=lbl.shape, 
                    n_frames=n_frames, 
                    device=self.device
                )
                
                # Forward pass
                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(feat, global_step=global_step)

                # Compute loss (criterion handles masking internally)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.clip_norm > 0.0:
                    clip_grad_norm_(net.parameters(), self.clip_norm)
                
                optimizer.step()
                
                # Accumulate loss
                running_loss = loss.data.item()
                accu_tr_loss += running_loss * n_frames_sum
                accu_n_frames += n_frames_sum

                iter_end_time = timeit.default_timer()
                batch_time = iter_end_time - iter_start_time
                
                current_lr = optimizer.param_groups[0]['lr']

                # Logging
                log_msg = f'Epoch [{ckpt_info["cur_epoch"] + 1}/{self.max_n_epochs}], ' \
                          f'Iter [{n_iter + 1}/{self.logging_period}], ' \
                          f'LR: {current_lr:.6f}, ' \
                          f'tr_loss: {running_loss:.4f} / {accu_tr_loss / accu_n_frames:.4f}, ' \
                          f'time: {batch_time:.4f}s'
                
                if self.time_log:
                    with open(self.time_log, 'a+') as f:
                        print(log_msg, file=f)
                        f.flush()
                else:
                    print(log_msg, flush=True)
            
            # End of epoch - run validation
            avg_tr_loss = accu_tr_loss / accu_n_frames
            
            logger.info('\n' + '='*70)
            logger.info(f'VALIDATION - Epoch {ckpt_info["cur_epoch"] + 1}/{self.max_n_epochs}')
            logger.info('='*70)
            
            # Run validation
            avg_cv_loss = self.validate(
                net, valid_loader, criterion, feeder, global_step, logger
            )
            
            # Restore training mode
            net.train()
            
            # Update checkpoint info
            ckpt_info['cur_iter'] = n_iter
            ckpt_info['global_step'] = global_step
            is_best = avg_cv_loss < ckpt_info['best_loss']
            
            if is_best:
                improvement = ckpt_info['best_loss'] - avg_cv_loss
                logger.info(f'NEW BEST MODEL! Improvement: {improvement:.4f}')
                ckpt_info['best_loss'] = avg_cv_loss
                min_lr_epoch_count = 0
            
            ckpt_info['tr_loss'] = avg_tr_loss
            ckpt_info['cv_loss'] = avg_cv_loss
            
            # ReduceLROnPlateau step
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_cv_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if abs(old_lr - new_lr) > 1e-8:
                logger.info('='*70)
                logger.info('LEARNING RATE REDUCED')
                logger.info(f'LR: {old_lr:.6f} -> {new_lr:.6f}')
                logger.info('='*70)
            
            # Early stopping logic
            if new_lr <= self.plateau_min_lr:
                if not is_best:
                    min_lr_epoch_count += 1
                    logger.info(f'Early stop counter: {min_lr_epoch_count}/{self.early_stop_patience}')
                    
                    if min_lr_epoch_count >= self.early_stop_patience:
                        logger.info('='*70)
                        logger.info('EARLY STOPPING TRIGGERED')
                        logger.info(f'No improvement for {self.early_stop_patience} epochs at min LR')
                        logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
                        logger.info('='*70)
                        
                        # Save final checkpoint
                        self._save_checkpoint(ckpt_info, net, optimizer, scheduler, is_best=False)
                        logger.info('Training stopped by early stopping.\n')
                        return
            else:
                min_lr_epoch_count = 0
            
            ckpt_info['min_lr_epoch_count'] = min_lr_epoch_count
            
            # Save checkpoint
            logger.info(f'Train Loss: {avg_tr_loss:.4f} | CV Loss: {avg_cv_loss:.4f} | '
                       f'Best: {ckpt_info["best_loss"]:.4f} | LR: {new_lr:.6f}')
            logger.info('='*70 + '\n')
            
            self._save_checkpoint(ckpt_info, net, optimizer, scheduler, is_best)
            
            # Write to loss log
            ckpt = CheckPoint(ckpt_info, None, None, None)
            lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, self.logging_period)

            # PESQ Validation (every N epochs)
            # APPROACH D: Smart Periodic - Test PESQ on best.pt model, not current model
            current_epoch = ckpt_info['cur_epoch'] + 1  # 1-indexed for display
            if (self.pesq_eval_interval > 0 and
                current_epoch % self.pesq_eval_interval == 0):

                logger.info('='*70)
                logger.info(f'PESQ VALIDATION - Epoch {current_epoch}')
                logger.info('='*70)

                # Load best model (by validation loss) for PESQ testing
                best_model_path = os.path.join(self.ckpt_dir, 'models', 'best.pt')

                if os.path.exists(best_model_path):
                    logger.info('Loading best model (by cv_loss) for PESQ evaluation...')

                    # Load best checkpoint
                    ckpt_best = CheckPoint()
                    ckpt_best.load(best_model_path, self.device)

                    # Create temporary model with best weights
                    net_best = Net(F=self.F).to(self.device)

                    # Load state dict (handle DataParallel case)
                    if len(self.gpu_ids) > 1:
                        # Add 'module.' prefix for DataParallel
                        state_dict = {}
                        for key in ckpt_best.net_state_dict:
                            state_dict['module.' + key] = ckpt_best.net_state_dict[key]
                        net_best = DataParallel(net_best, device_ids=self.gpu_ids)
                        net_best.load_state_dict(state_dict)
                    else:
                        net_best.load_state_dict(ckpt_best.net_state_dict)

                    best_epoch = ckpt_best.ckpt_info['cur_epoch'] + 1
                    best_cv_loss = ckpt_best.ckpt_info['best_loss']
                    logger.info(f'Testing best model from epoch {best_epoch} '
                               f'(cv_loss={best_cv_loss:.4f})')

                    # Test PESQ on best model (not current model!)
                    pesq_start_time = timeit.default_timer()
                    avg_pesq = self.validate_pesq(
                        net_best, valid_loader, feeder, resynthesizer, logger
                    )
                    pesq_eval_time = timeit.default_timer() - pesq_start_time

                    # Check if best PESQ
                    is_best_pesq = avg_pesq > best_pesq
                    if is_best_pesq:
                        improvement = avg_pesq - best_pesq
                        logger.info(f'NEW BEST PESQ! {best_pesq:.4f} -> {avg_pesq:.4f} (+{improvement:.4f})')
                        best_pesq = avg_pesq
                        ckpt_info['best_pesq'] = best_pesq

                        # Save best PESQ model (use the loaded best model, not current net)
                        if self.save_best_pesq_model:
                            model_path = os.path.join(self.ckpt_dir, 'models')
                            pesq_model_path = os.path.join(model_path, 'best_pesq.pt')
                            # Save the best model we just tested
                            torch.save(ckpt_best, pesq_model_path)
                            logger.info(f'Saved best PESQ model to: {pesq_model_path}')
                            logger.info(f'  (Model from epoch {best_epoch} with cv_loss={best_cv_loss:.4f})')

                    # Log PESQ results
                    pesqLog(
                        os.path.join(self.ckpt_dir, self.pesq_log),
                        current_epoch, avg_pesq, best_pesq,
                        len(valid_loader.dataset), pesq_eval_time
                    )
                    logger.info(f'PESQ on best.pt (epoch {best_epoch}): {avg_pesq:.4f} | '
                               f'Best PESQ overall: {best_pesq:.4f} | Time: {pesq_eval_time:.1f}s')
                    logger.info('='*70 + '\n')

                    # Clean up temporary model
                    del net_best

                else:
                    logger.warning(f'Best model not found at {best_model_path}')
                    logger.warning('Skipping PESQ evaluation (will try next interval)')
                    logger.info('='*70 + '\n')

            # Next epoch
            ckpt_info['cur_epoch'] += 1
        
        logger.info('='*70)
        logger.info('TRAINING COMPLETED')
        logger.info(f'Total epochs: {ckpt_info["cur_epoch"]}')
        logger.info(f'Best CV loss: {ckpt_info["best_loss"]:.4f}')
        if best_pesq > 0:
            logger.info(f'Best PESQ: {best_pesq:.4f}')
        logger.info('='*70)

        return
    
    def _save_checkpoint(self, ckpt_info, net, optimizer, scheduler, is_best):
        """Helper to save checkpoint"""
        model_path = os.path.join(self.ckpt_dir, 'models')
        os.makedirs(model_path, exist_ok=True)
        
        if len(self.gpu_ids) > 1:
            ckpt = CheckPoint(
                ckpt_info, 
                net.module.state_dict(), 
                optimizer.state_dict(),
                scheduler.state_dict()
            )
        else:
            ckpt = CheckPoint(
                ckpt_info, 
                net.state_dict(), 
                optimizer.state_dict(),
                scheduler.state_dict()
            )
        
        ckpt.save(
            os.path.join(model_path, 'latest.pt'),
            is_best,
            os.path.join(model_path, 'best.pt') if is_best else None
        )

    def validate(self, net, cv_loader, criterion, feeder, global_step, logger):
        """Validation procedure"""
        accu_cv_loss = 0.
        accu_n_frames = 0

        model = net.module if isinstance(net, DataParallel) else net
        model.eval()
        
        with torch.no_grad():
            for batch in cv_loader:
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples'].to(self.device)
                
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)

                feat, lbl = feeder(mix, sph)
                loss_mask = lossMask(
                    shape=lbl.shape,
                    n_frames=n_frames,
                    device=self.device
                )

                est = model(feat, global_step=global_step)
                # Compute loss (criterion handles masking internally)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)

                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)
                
                accu_cv_loss += loss.data.item() * n_frames_sum
                accu_n_frames += n_frames_sum
        
        avg_cv_loss = accu_cv_loss / accu_n_frames
        return avg_cv_loss

    def validate_pesq(self, net, cv_loader, feeder, resynthesizer, logger):
        """
        PESQ Validation on full validation set
        Returns average PESQ score across all samples
        """
        if not PESQ_AVAILABLE:
            logger.warning('PESQ not available. Skipping PESQ validation.')
            return -1.0

        model = net.module if isinstance(net, DataParallel) else net
        model.eval()

        pesq_scores = []
        n_samples_processed = 0
        n_errors = 0

        logger.info('Computing PESQ on validation set...')

        with torch.no_grad():
            for batch_idx, batch in enumerate(cv_loader):
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples']

                # Forward pass
                feat, lbl = feeder(mix, sph)
                est = model(feat, global_step=None)

                # Resynthesize audio
                sph_est = resynthesizer(est, mix)

                # Process each sample in batch
                batch_size = mix.shape[0]
                for i in range(batch_size):
                    # Get actual length
                    if isinstance(n_samples, torch.Tensor):
                        if n_samples.ndim == 0:
                            actual_len = n_samples.item()
                        else:
                            actual_len = n_samples[i].item()
                    else:
                        actual_len = int(n_samples)

                    # Get clean and enhanced audio, trim to actual length
                    clean_audio = sph[i].cpu().numpy()[:actual_len]
                    est_audio = sph_est[i].cpu().numpy()[:actual_len]

                    # Compute PESQ (wideband mode for 16kHz)
                    try:
                        score = pesq(self.sample_rate, clean_audio, est_audio, 'wb')
                        pesq_scores.append(score)
                        n_samples_processed += 1
                    except Exception as e:
                        n_errors += 1
                        if n_errors <= 3:  # Only log first 3 errors
                            logger.warning(f'PESQ error on sample {n_samples_processed}: {e}')

                # Progress logging every 50 batches
                if (batch_idx + 1) % 50 == 0:
                    logger.info(f'  Processed {n_samples_processed} samples...')

        if len(pesq_scores) == 0:
            logger.error('No valid PESQ scores computed!')
            return -1.0

        avg_pesq = np.mean(pesq_scores)
        logger.info(f'PESQ Results: avg={avg_pesq:.4f}, samples={n_samples_processed}, errors={n_errors}')

        return avg_pesq

    def test(self):
        """
        Testing procedure - WITH PADDING FIX
        Trims audio to actual length before saving to avoid padding artifacts in metrics
        """
        print("\n" + "="*70)
        print("STEP 1: VALIDATING TEST DATA")
        print("="*70)
        try:
            validate_data_dirs(mode='test')
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? DATA VALIDATION FAILED: {e}")
            print("\nPlease check your configs.py paths!")
            return
        
        # Load configuration
        self.model_file = test_conf['model_file']
        self.ckpt_dir = train_conf['ckpt_dir']
        self.est_path = train_conf['est_path']
        self.write_ideal = test_conf['write_ideal']
        
        # Setup device
        self.gpu_ids = tuple(map(int, train_conf['gpu_ids'].split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{self.gpu_ids[0]}')

        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.est_path, exist_ok=True)
        
        print("\n" + "="*70)
        print("STEP 2: INITIALIZING LOGGER")
        print("="*70)
        
        # Setup logger
        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)
        
        logger.info('='*70)
        logger.info('TESTING PROCEDURE (WITH PADDING FIX)')
        logger.info('='*70)
        logger.info(f'Model file: {self.model_file}')
        logger.info(f'Test clean: {TEST_CLEAN_DIR}')
        logger.info(f'Test noisy: {TEST_NOISY_DIR}')
        logger.info(f'Output path: {self.est_path}')
        logger.info(f'Device: {self.device}')
        logger.info(f'? PADDING FIX: Enabled - will trim to actual length before saving')
        logger.info('='*70 + '\n')

        print("\n" + "="*70)
        print("STEP 3: LOADING MODEL")
        print("="*70)

        # Create network with correct frequency bins
        logger.info(f'STFT config: win_size={self.win_size}, hop_size={self.hop_size}, F={self.F}')
        net = Net(F=self.F)
        net = net.to(self.device)

        param_count = numParams(net)
        logger.info(f'Parameters: {param_count:,d} -> {param_count*32/8/(2**20):.2f} MB\n')
        
        # Create utilities
        criterion = LossFunction(device=self.device, win_size=self.win_size, hop_size=self.hop_size)
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)
        
        # Load model
        logger.info(f'Loading model from: {self.model_file}')
        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)
        
        logger.info(f'Model loaded successfully!')
        logger.info(f'  Epoch: {ckpt.ckpt_info["cur_epoch"] + 1}')
        logger.info(f'  Best loss: {ckpt.ckpt_info["best_loss"]:.4f}\n')
        
        net.eval()
        
        print("\n" + "="*70)
        print("STEP 4: CREATING TEST DATALOADER")
        print("="*70)
        
        # Create test loader
        cache_dir = os.path.join(self.ckpt_dir, 'cache')
        
        test_loader = create_test_dataloader_only(
            test_clean_dir=TEST_CLEAN_DIR,
            test_noisy_dir=TEST_NOISY_DIR,
            batch_size=test_conf['batch_size'],
            num_workers=test_conf['num_workers'],
            sample_rate=self.sample_rate,
            unit='seg',
            segment_size=6.0,
            segment_shift=6.0,
            max_length_seconds=6.0,
            pin_memory=True,
            cache_dir=cache_dir
        )
        
        print("\n" + "="*70)
        print("STEP 5: PROCESSING TEST SET (WITH PADDING FIX)")
        print("="*70 + "\n")
        
        # Process test set
        accu_tt_loss = 0.
        accu_n_frames = 0
        
        # Statistics tracking
        total_samples = 0
        trimmed_samples = 0
        skipped_samples = 0
        
        with torch.no_grad():
            for k, batch in enumerate(test_loader):
                mix = batch['mix'].to(self.device)
                sph = batch['sph'].to(self.device)
                n_samples = batch['n_samples']

                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                
                feat, lbl = feeder(mix, sph)
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)

                est = net(feat, global_step=None)
                # FIX: Mask output to match training behavior (prevents noise in padded regions)
                est = est * loss_mask
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)
                
                if isinstance(n_frames, torch.Tensor):
                    n_frames_sum = n_frames.sum().item()
                else:
                    n_frames_sum = sum(n_frames)
                
                accu_tt_loss += loss.data.item() * n_frames_sum
                accu_n_frames += n_frames_sum
                
                # Resynthesize audio
                sph_idl = resynthesizer(lbl, mix)
                sph_est = resynthesizer(est, mix)
                
                # Convert to numpy
                mix_np = mix[0].cpu().numpy()
                sph_np = sph[0].cpu().numpy()
                sph_est_np = sph_est[0].cpu().numpy()
                sph_idl_np = sph_idl[0].cpu().numpy()
                
                # ??? CRITICAL FIX: Get actual length and validate
                if isinstance(n_samples, torch.Tensor):
                    if n_samples.ndim == 0:
                        n_samples_value = n_samples.item()
                    else:
                        n_samples_value = n_samples[0].item()
                else:
                    n_samples_value = int(n_samples)
                
                actual_length = len(mix_np)
                
                # Validate n_samples_value
                if n_samples_value <= 0:
                    logger.error(
                        f'Sample {k+1} ({batch["filenames"][0]}): '
                        f'n_samples={n_samples_value} is invalid. Skipping...'
                    )
                    skipped_samples += 1
                    continue
                
                if n_samples_value > actual_length:
                    logger.warning(
                        f'Sample {k+1} ({batch["filenames"][0]}): '
                        f'n_samples={n_samples_value} > actual_length={actual_length}. '
                        f'Using actual_length.'
                    )
                    n_samples_value = actual_length
                
                # ??? CRITICAL FIX: Trim to actual length (removes padding artifacts)
                if n_samples_value < actual_length:
                    mix_np = mix_np[:n_samples_value]
                    sph_np = sph_np[:n_samples_value]
                    sph_est_np = sph_est_np[:n_samples_value]
                    sph_idl_np = sph_idl_np[:n_samples_value]
                    trimmed_samples += 1
                    
                    # Log first trimmed sample for verification
                    if trimmed_samples == 1:
                        logger.info(
                            f'Sample {k+1}: Trimmed from {actual_length} to {n_samples_value} samples '
                            f'(removed {actual_length - n_samples_value} padding samples)'
                        )
                
                # Final safety check
                if len(mix_np) == 0:
                    logger.error(
                        f'Sample {k+1} ({batch["filenames"][0]}): '
                        f'Empty audio after trimming. Skipping...'
                    )
                    skipped_samples += 1
                    continue
                
                # NOW normalize (after trimming and validation)
                mix_np, sph_np, sph_est_np, sph_idl_np = wavNormalize(
                    mix_np, sph_np, sph_est_np, sph_idl_np
                )
                
                # Get filename from batch
                filename_base = os.path.splitext(batch['filenames'][0])[0]
                
                # Save trimmed audio files (no padding artifacts!)
                sf.write(
                    os.path.join(self.est_path, f'{filename_base}_mix.wav'), 
                    mix_np, self.sample_rate
                )
                sf.write(
                    os.path.join(self.est_path, f'{filename_base}_sph.wav'), 
                    sph_np, self.sample_rate
                )
                sf.write(
                    os.path.join(self.est_path, f'{filename_base}_sph_est.wav'), 
                    sph_est_np, self.sample_rate
                )
                
                if self.write_ideal:
                    sf.write(
                        os.path.join(self.est_path, f'{filename_base}_sph_idl.wav'), 
                        sph_idl_np, self.sample_rate
                    )
                
                total_samples += 1
                
                if (k + 1) % 10 == 0:
                    logger.info(f'  Processed {k + 1} samples')

        avg_tt_loss = accu_tt_loss / accu_n_frames
        
        # Log final statistics
        logger.info(f'\n{"="*70}')
        logger.info('TEST RESULTS')
        logger.info(f'{"="*70}')
        logger.info(f'Average test loss: {avg_tt_loss:.4f}')
        logger.info(f'Total samples processed: {total_samples}')
        logger.info(f'Samples with padding trimmed: {trimmed_samples} ({100*trimmed_samples/max(total_samples,1):.1f}%)')
        logger.info(f'Samples skipped (errors): {skipped_samples}')
        logger.info(f'Output directory: {self.est_path}')
        logger.info('='*70)
        logger.info('? TESTING COMPLETED')
        logger.info('? Audio trimmed to actual lengths (padding removed)')
        logger.info('? PESQ/STOI metrics should now be accurate and fair')
        logger.info('='*70)
       
        return


if __name__ == '__main__':
    import sys
    
    model = Model()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            model.train()
        elif sys.argv[1] == 'test':
            model.test()
        else:
            print(f'? Unknown command: {sys.argv[1]}')
            print('Usage: python model.py [train|test]')
    else:
        print('Usage: python model.py [train|test]')