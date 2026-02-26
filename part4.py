import os
import random
import glob
import math
import numpy as np
import soundfile as sf
from scipy import signal

# ==========================================
# 1. CONFIGURATION (ID Sum = 0)
# ==========================================

# Condition 0: Strong Noise (0-6 dB)
# We use environmental noise (not music)
NOISE_TYPE = "noise" 
MIN_SNR = 0
MAX_SNR = 6

# Paths
# Path to the extracted MUSAN folder
MUSAN_PATH = "musan"  

# Path to the Common Voice clips
# (Update this if your folder name is different)
CV_CLIPS_PATH = os.path.join("cv-corpus-24.0-2025-12-05-he.tar", "cv-corpus-24.0-2025-12-05", "he", "clips")

# Output directory for the augmented files
OUTPUT_DIR = "noisy_clips"

# Log file to record what noise was added to which file
LOG_FILE = "augmentation_log.tsv"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_valid_noise_files():
    """
    Scans the MUSAN directory for valid noise files.
    Criteria:
    1. Must be in 'free-sound' or 'sound-bible' folders (environmental noise).
    2. Must be longer than 30 seconds (to avoid short sounds like door slams).
    """
    print(f"Scanning {MUSAN_PATH} for valid noise files...")
    valid_files = []
    
    # Recursively find all .wav files in the noise folder
    search_path = os.path.join(MUSAN_PATH, "noise", "**", "*.wav")
    all_files = glob.glob(search_path, recursive=True)
    
    for f in all_files:
        # Filter for environmental noise folders
        if "free-sound" in f or "sound-bible" in f:
            try:
                # Check duration
                info = sf.info(f)
                if info.duration >= 30:
                    valid_files.append(f)
            except Exception as e:
                print(f"Warning: Could not read {f}. Skipping.")
                continue
                
    print(f"Found {len(valid_files)} valid noise files (>30s).")
    return valid_files

def calculate_power(signal_data):
    """
    Calculates the average power of a signal (Mean Square).
    """
    return np.mean(signal_data ** 2)

def add_noise_to_speech(speech, noise, target_snr):
    """
    Mixes the speech signal with the noise signal at the specified SNR.
    
    Formula:
    SNR (dB) = 10 * log10(P_speech / P_noise_scaled)
    We need to find 'alpha' to scale the noise:
    alpha = sqrt( P_speech / ( 10^(SNR/10) * P_noise ) )
    """
    p_speech = calculate_power(speech)
    p_noise = calculate_power(noise)
    
    # Avoid division by zero
    if p_noise == 0:
        return speech
    
    # Calculate the target ratio based on dB
    target_ratio = 10 ** (target_snr / 10)
    
    # Calculate the scaling factor (alpha)
    alpha = math.sqrt(p_speech / (target_ratio * p_noise))
    
    # Mix the signals
    noisy_speech = speech + (alpha * noise)
    
    # Normalize the audio to prevent clipping (values > 1.0 or < -1.0)
    max_val = np.max(np.abs(noisy_speech))
    if max_val > 1.0:
        noisy_speech = noisy_speech / max_val
        
    return noisy_speech

# ==========================================
# 3. MAIN PROCESSING LOOP
# ==========================================

def main():
    # 1. Setup directories and lists
    if not os.path.exists(CV_CLIPS_PATH):
        print(f"Error: Could not find clips folder at: {CV_CLIPS_PATH}")
        print("Please check the folder name in the configuration section.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    noise_files = get_valid_noise_files()
    if not noise_files:
        print("Error: No noise files found. Please ensure 'musan' is extracted correctly.")
        return

    # Get list of speech files (MP3)
    speech_files = glob.glob(os.path.join(CV_CLIPS_PATH, "*.mp3"))
    print(f"Found {len(speech_files)} speech files to process.")

    # 2. Open Log File
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        # Write Header
        log.write("Filename\tBackground file\tStart point (s)\tSNR\n")
        
        # 3. Iterate over files
        for i, speech_path in enumerate(speech_files):
            filename = os.path.basename(speech_path)
            
            try:
                # A. Read Speech Audio
                speech_data, sample_rate = sf.read(speech_path)
                
                # B. Decimate (Downsample)
                # Assignment requirement: Use scipy.signal.decimate with factor 2.
                # This simulates lower quality audio and matches MUSAN's 16kHz rate better.
                if sample_rate > 16000:
                    # 'q=2' means downsample by factor of 2 (e.g., 32k -> 16k)
                    speech_data = signal.decimate(speech_data, q=2)
                    sample_rate = int(sample_rate / 2)

                # C. Select Random Noise File
                noise_path = random.choice(noise_files)
                noise_info = sf.info(noise_path)
                
                # D. Select Random Start Point in Noise File
                # We need a slice of noise that is exactly the same length as the speech
                speech_len_samples = len(speech_data)
                noise_len_samples = noise_info.frames
                
                # Calculate maximum valid start point
                max_start = noise_len_samples - speech_len_samples
                
                if max_start <= 0:
                    # If noise is shorter than speech (rare for >30s files), skip this iteration
                    continue
                
                start_sample = random.randint(0, max_start)
                start_seconds = start_sample / noise_info.samplerate
                
                # Read the specific slice of noise
                # Note: Common Voice is usually mono. If Musan is stereo, sf.read might return 2 channels.
                noise_data, _ = sf.read(noise_path, start=start_sample, frames=speech_len_samples)
                
                # Ensure noise is mono (if it has 2 dimensions, average them)
                if len(noise_data.shape) > 1:
                    noise_data = np.mean(noise_data, axis=1)

                # E. Mix signals
                # Random SNR between 0 and 6 dB
                current_snr = random.uniform(MIN_SNR, MAX_SNR)
                augmented_audio = add_noise_to_speech(speech_data, noise_data, current_snr)
                
                # F. Save to Output Directory
                # We save as .wav because it's easier to handle than re-encoding mp3
                output_filename = filename.replace(".mp3", ".wav")
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                sf.write(output_path, augmented_audio, sample_rate)
                
                # G. Write to Log
                log.write(f"{filename}\t{os.path.basename(noise_path)}\t{start_seconds:.2f}\t{current_snr:.2f}\n")
                
                # Print progress every 100 files
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(speech_files)} files...")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("==========================================")
    print(f"Done! Noisy files saved in: {OUTPUT_DIR}")
    print(f"Log file created: {LOG_FILE}")
    print("==========================================")

if __name__ == "__main__":
    main()