import os
import soundfile as sf
from eval_utils import CompositeEval
import numpy as np


def save_scores_to_txt(filename, file_names, csig_enh_list, cbak_enh_list, 
                       covl_enh_list, ssnr_enh_list, pesq_enh_list=None):
    """
    Save individual scores to text file
    
    Args:
        filename: Output filename
        file_names: List of file names
        csig_enh_list: List of CSIG scores
        cbak_enh_list: List of CBAK scores
        covl_enh_list: List of COVL scores
        ssnr_enh_list: List of SSNR scores
        pesq_enh_list: Optional list of PESQ scores
    """
    with open(filename, "w") as file:
        if pesq_enh_list is not None:
            for fname, csig, cbak, covl, ssnr, pesq_val in zip(
                file_names, csig_enh_list, cbak_enh_list, covl_enh_list, 
                ssnr_enh_list, pesq_enh_list):
                file.write("File: {}\n".format(fname))
                file.write("CSIG: {:.4f}\n".format(csig))
                file.write("CBAK: {:.4f}\n".format(cbak))
                file.write("COVL: {:.4f}\n".format(covl))
                file.write("SSNR: {:.4f}\n".format(ssnr))
                file.write("PESQ: {:.4f}\n".format(pesq_val))
                file.write("\n")
        else:
            for fname, csig, cbak, covl, ssnr in zip(
                file_names, csig_enh_list, cbak_enh_list, covl_enh_list, ssnr_enh_list):
                file.write("File: {}\n".format(fname))
                file.write("CSIG: {:.4f}\n".format(csig))
                file.write("CBAK: {:.4f}\n".format(cbak))
                file.write("COVL: {:.4f}\n".format(covl))
                file.write("SSNR: {:.4f}\n".format(ssnr))
                file.write("\n")


def detect_sample_rate(file_path):
    """
    Detect sample rate from audio file
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Sample rate (int)
    """
    try:
        info = sf.info(file_path)
        return info.samplerate
    except Exception as e:
        print("Warning: Could not detect sample rate from {}: {}".format(file_path, e))
        return 16000  # Default fallback


def process_subfolder(subfolder_path, all_scores_dir):
    """
    Process all files in a subfolder and compute composite metrics
    
    FIXED: 
    - Proper NaN handling in averages
    - Sample rate detection
    - Better error handling
    - Epsilon protection for divisions
    
    Args:
        subfolder_path: Path to subfolder containing audio files
        all_scores_dir: Directory to save scores
    """
    clean_files = []
    enhanced_files = []

    subfolder_name = os.path.basename(subfolder_path)
    print("\n" + "="*60)
    print("Processing subfolder: {}".format(subfolder_name))
    print("="*60)

    # Collect all clean and enhanced files
    for root, dirs, files in os.walk(subfolder_path):
        for file_name in files:
            if file_name.endswith("_sph.wav"):
                clean_files.append(os.path.join(root, file_name))
            elif file_name.endswith("_sph_est.wav"):
                enhanced_files.append(os.path.join(root, file_name))

    if len(clean_files) == 0:
        print("Warning: No clean files found in {}".format(subfolder_name))
        return

    # Detect sample rate from first file
    srate = detect_sample_rate(clean_files[0])
    print("Detected sample rate: {} Hz".format(srate))

    # Initialize score lists
    file_names = []
    csig_enh_list = []
    cbak_enh_list = []
    covl_enh_list = []
    ssnr_enh_list = []
    pesq_enh_list = []
    
    processed_count = 0
    skipped_count = 0
    error_files = []

    for clean_file in clean_files:
        clean_name = os.path.splitext(clean_file)[0]
        enhanced_file = next(
            (f for f in enhanced_files if os.path.splitext(f)[0] == clean_name + "_est"), 
            None
        )

        if enhanced_file is None:
            print("Warning: No matching enhanced file for {}".format(os.path.basename(clean_file)))
            skipped_count += 1
            continue

        try:
            # Read audio files
            clean_wav, sr_clean = sf.read(clean_file, dtype='float32')
            enh_wav, sr_enh = sf.read(enhanced_file, dtype='float32')
            
            # Validate sample rates
            if sr_clean != sr_enh:
                print("Warning: Sample rate mismatch in {}: clean={}, enhanced={}".format(
                    os.path.basename(clean_file), sr_clean, sr_enh))
                skipped_count += 1
                continue
            
            if sr_clean != srate:
                print("Warning: Sample rate mismatch in {}: expected={}, got={}".format(
                    os.path.basename(clean_file), srate, sr_clean))
                srate = sr_clean  # Update to actual rate
            
            # Align lengths
            min_len = min(clean_wav.shape[0], enh_wav.shape[0])
            if min_len == 0:
                print("Warning: Empty audio file {}".format(os.path.basename(clean_file)))
                skipped_count += 1
                continue
            
            clean_wav = clean_wav[:min_len]
            enh_wav = enh_wav[:min_len]
            
            # Validate signals are not all zeros
            if np.allclose(clean_wav, 0) or np.allclose(enh_wav, 0):
                print("Warning: All-zero audio in {}".format(os.path.basename(clean_file)))
                skipped_count += 1
                continue

            # Compute composite metrics
            # FIXED: Pass sample rate to CompositeEval
            csig_enh, cbak_enh, covl_enh, pesq_enh, ssnr_enh = CompositeEval(
                clean_wav, enh_wav, srate=srate, log_all=True
            )

            # FIXED: Proper NaN validation
            if (np.isnan(csig_enh) or np.isnan(cbak_enh) or 
                np.isnan(covl_enh) or np.isnan(ssnr_enh) or np.isnan(pesq_enh)):
                print("Warning: NaN values in {}".format(os.path.basename(clean_file)))
                error_files.append(os.path.basename(clean_file))
                skipped_count += 1
                continue
            
            # Check for infinite values
            if (np.isinf(csig_enh) or np.isinf(cbak_enh) or 
                np.isinf(covl_enh) or np.isinf(ssnr_enh) or np.isinf(pesq_enh)):
                print("Warning: Inf values in {}".format(os.path.basename(clean_file)))
                error_files.append(os.path.basename(clean_file))
                skipped_count += 1
                continue

            # Store valid scores
            file_names.append(os.path.basename(clean_file))
            csig_enh_list.append(csig_enh)
            cbak_enh_list.append(cbak_enh)
            covl_enh_list.append(covl_enh)
            ssnr_enh_list.append(ssnr_enh)
            pesq_enh_list.append(pesq_enh)
            processed_count += 1

        except Exception as e:
            print("Error processing {}: {}".format(os.path.basename(clean_file), str(e)))
            error_files.append(os.path.basename(clean_file))
            skipped_count += 1
            continue

    # Print summary
    print("\nProcessing summary for {}:".format(subfolder_name))
    print("  Processed: {}".format(processed_count))
    print("  Skipped: {}".format(skipped_count))
    if error_files:
        show_files = error_files[:5]
        print("  Error files: {}{}".format(
            show_files,
            " ... and {} more".format(len(error_files)-5) if len(error_files) > 5 else ""
        ))

    # FIXED: Calculate averages only if we have valid scores
    if processed_count == 0:
        print("Warning: No valid files processed in {}".format(subfolder_name))
        return

    # Compute averages (all values are valid, no NaN)
    avg_csig = np.mean(csig_enh_list)
    avg_cbak = np.mean(cbak_enh_list)
    avg_covl = np.mean(covl_enh_list)
    avg_ssnr = np.mean(ssnr_enh_list)
    avg_pesq = np.mean(pesq_enh_list)

    # Compute standard deviations
    std_csig = np.std(csig_enh_list)
    std_cbak = np.std(cbak_enh_list)
    std_covl = np.std(covl_enh_list)
    std_ssnr = np.std(ssnr_enh_list)
    std_pesq = np.std(pesq_enh_list)

    # Create output directory
    subfolder_scores_dir = os.path.join(all_scores_dir, subfolder_name)
    if not os.path.exists(subfolder_scores_dir):
        os.makedirs(subfolder_scores_dir)

    # Save average scores
    avg_scores_filename = os.path.join(subfolder_scores_dir, 
                                       "{}_average_scores.txt".format(subfolder_name))
    with open(avg_scores_filename, "w") as file:
        file.write("Subfolder: {}\n".format(subfolder_name))
        file.write("Sample Rate: {} Hz\n".format(srate))
        file.write("Processed Files: {}\n".format(processed_count))
        file.write("Skipped Files: {}\n".format(skipped_count))
        file.write("\n{}\n".format("="*40))
        file.write("Average CSIG: {:.4f} +/- {:.4f}\n".format(avg_csig, std_csig))
        file.write("Average CBAK: {:.4f} +/- {:.4f}\n".format(avg_cbak, std_cbak))
        file.write("Average COVL: {:.4f} +/- {:.4f}\n".format(avg_covl, std_covl))
        file.write("Average SSNR: {:.4f} +/- {:.4f}\n".format(avg_ssnr, std_ssnr))
        file.write("Average PESQ: {:.4f} +/- {:.4f}\n".format(avg_pesq, std_pesq))

    # Save individual scores
    individual_scores_filename = os.path.join(subfolder_scores_dir, 
                                              "{}_scores.txt".format(subfolder_name))
    save_scores_to_txt(individual_scores_filename, file_names, 
                       csig_enh_list, cbak_enh_list, covl_enh_list, 
                       ssnr_enh_list, pesq_enh_list)

    # Print results
    print("\nResults for {}:".format(subfolder_name))
    print("  CSIG: {:.4f} +/- {:.4f}".format(avg_csig, std_csig))
    print("  CBAK: {:.4f} +/- {:.4f}".format(avg_cbak, std_cbak))
    print("  COVL: {:.4f} +/- {:.4f}".format(avg_covl, std_covl))
    print("  SSNR: {:.4f} +/- {:.4f}".format(avg_ssnr, std_ssnr))
    print("  PESQ: {:.4f} +/- {:.4f}".format(avg_pesq, std_pesq))

    # Append to main summary file
    summary_file = os.path.join(all_scores_dir, "..", "Average-CSIG_CBAK_COVL_SSNR_PESQ.txt")
    with open(summary_file, "a") as file:
        file.write("\n{}\n".format("="*60))
        file.write("Subfolder: {}\n".format(subfolder_name))
        file.write("Sample Rate: {} Hz\n".format(srate))
        file.write("Processed Files: {}\n".format(processed_count))
        file.write("{}\n".format("="*60))
        file.write("Average CSIG: {:.4f} +/- {:.4f}\n".format(avg_csig, std_csig))
        file.write("Average CBAK: {:.4f} +/- {:.4f}\n".format(avg_cbak, std_cbak))
        file.write("Average COVL: {:.4f} +/- {:.4f}\n".format(avg_covl, std_covl))
        file.write("Average SSNR: {:.4f} +/- {:.4f}\n".format(avg_ssnr, std_ssnr))
        file.write("Average PESQ: {:.4f} +/- {:.4f}\n".format(avg_pesq, std_pesq))


def main(est_path, ckpt_dir):
    """
    Main function to process all subfolders
    
    Args:
        est_path: Path to estimates directory
        ckpt_dir: Path to checkpoint directory for saving results
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    all_scores_dir = os.path.join(ckpt_dir, 'all_scores')
    if not os.path.exists(all_scores_dir):
        os.makedirs(all_scores_dir)

    # Clear or create summary file
    summary_file = os.path.join(ckpt_dir, "Average-CSIG_CBAK_COVL_SSNR_PESQ.txt")
    with open(summary_file, "w") as file:
        file.write("COMPOSITE METRICS EVALUATION\n")
        file.write("{}\n\n".format("="*60))

    # Process each subfolder
    subfolders = [d for d in os.listdir(est_path) 
                  if os.path.isdir(os.path.join(est_path, d))]
    
    print("\nFound {} subfolders to process".format(len(subfolders)))
    
    for subfolder_name in sorted(subfolders):
        subfolder_path = os.path.join(est_path, subfolder_name)
        process_subfolder(subfolder_path, all_scores_dir)

    print("\n{}".format("="*60))
    print("Processing complete!")
    print("Results saved to: {}".format(ckpt_dir))
    print("{}\n".format("="*60))


if __name__ == "__main__":
    # Example usage
    est_path = '/path/to/estimates'
    ckpt_dir = '/path/to/checkpoint/dir'
    main(est_path, ckpt_dir) 