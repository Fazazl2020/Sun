"""
train.py - Standalone Training Script
Automatically starts training when run
No command-line arguments needed
"""

import sys
import os

# Ensure proper imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from utils.models import Model
from configs import validate_data_dirs


def main():
    """
    Main training function
    Validates data, creates model, and starts training
    """
    
    print("\n" + "="*70)
    print("SPEECH ENHANCEMENT MODEL - TRAINING")
    print("="*70)
    print("Project: T71a1")
    print("Mode: Training")
    print("="*70 + "\n")
    
    try:
        # Step 1: Validate data directories
        print("="*70)
        print("STEP 1: VALIDATING DATA DIRECTORIES")
        print("="*70)
        
        try:
            validate_data_dirs(mode='train')  # Validate train + valid data
        except (FileNotFoundError, ValueError) as e:
            print(f"\n? DATA VALIDATION FAILED!")
            print(f"Error: {e}")
            print("\nPlease check your configs.py paths:")
            print("  - DATASET_ROOT")
            print("  - TRAIN_CLEAN_DIR / TRAIN_NOISY_DIR")
            print("  - VALID_CLEAN_DIR / VALID_NOISY_DIR")
            sys.exit(1)
        
        # Step 2: Create model
        print("\n" + "="*70)
        print("STEP 2: INITIALIZING MODEL")
        print("="*70)
        
        model = Model()
        
        # Step 3: Start training
        print("\n" + "="*70)
        print("STEP 3: STARTING TRAINING")
        print("="*70)
        print("Press Ctrl+C to stop training")
        print("="*70 + "\n")
        
        model.train()
        
        # Training completed
        print("\n" + "="*70)
        print("? TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print("Check logs in: CHECKPOINT_DIR/logs/")
        print("Models saved in: CHECKPOINT_DIR/models/")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("??  TRAINING INTERRUPTED BY USER")
        print("="*70)
        print("Latest checkpoint saved in: CHECKPOINT_DIR/models/latest.pt")
        print("You can resume training by setting:")
        print("  TRAINING_CONFIG['resume_model'] = 'path/to/latest.pt'")
        print("in configs.py")
        print("="*70 + "\n")
        sys.exit(0)
        
    except Exception as e:
        print("\n\n" + "="*70)
        print("? ERROR OCCURRED DURING TRAINING")
        print("="*70)
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {e}\n")
        
        print("="*70)
        print("FULL TRACEBACK:")
        print("="*70)
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        
        sys.exit(1)


if __name__ == '__main__':
    main()