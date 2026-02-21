import os
import time
from faster_whisper import WhisperModel
from tqdm import tqdm

# --- CONFIGURATION ---
# UPDATE THIS PATH if necessary
DATASET_PATH = r"C:\Users\Maya\OneDrive\Desktop\Masters\asr_final_proj\cv-corpus-24.0-2025-12-05-he.tar\cv-corpus-24.0-2025-12-05\he"
CLIPS_FOLDER = os.path.join(DATASET_PATH, "clips")
INPUT_TEST_TSV = os.path.join(DATASET_PATH, "test.tsv")

OUTPUT_FILE = "results_part_a.tsv"
MODEL_ID = "ivrit-ai/whisper-large-v3-turbo-ct2"

def clean_filename(filename):
    """Removes extension: 'file.mp3' -> 'file'"""
    return os.path.splitext(os.path.basename(filename))[0]

def get_processed_files(output_file):
    """Reads the output file to find which clips are already done."""
    processed = set()
    if not os.path.exists(output_file):
        return processed
        
    print(f"Found existing results file: {output_file}")
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                # The filename is the first column
                processed.add(parts[0])
    
    print(f"--> Resuming! Found {len(processed)} clips already processed.")
    return processed

def run_transcription_resumable():
    print(f"--- STARTING RESUMABLE TRANSCRIPTION ---")
    
    # 1. Load Model
    print(f"Loading model: {MODEL_ID}...")
    try:
        model = WhisperModel(MODEL_ID, device="auto", compute_type="int8")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Check what is already done
    processed_files = get_processed_files(OUTPUT_FILE)

    # 3. Read Input File
    print(f"Reading input file: {INPUT_TEST_TSV}")
    with open(INPUT_TEST_TSV, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    data_lines = lines[1:]
    
    # 4. Open Output File in APPEND Mode ('a')
    # If file doesn't exist, 'a' creates it.
    mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    
    print(f"Writing to {OUTPUT_FILE} (Mode: {mode})")
    
    with open(OUTPUT_FILE, mode, encoding='utf-8', buffering=1) as f_out:
        
        # If we just created the file, write the header
        if mode == 'w':
            f_out.write("Filename\tReference Text\tTranscribed Text\n")
        
        # Loop with progress bar
        # We wrap data_lines in tqdm to show progress
        for line in tqdm(data_lines, unit="clip"):
            parts = line.strip().split('\t')
            
            if len(parts) <= 3:
                continue
                
            # Column Indices (Verified earlier): [1]=Path, [3]=Text
            audio_filename = parts[1]
            reference_text = parts[3]
            
            # --- CHECK IF DONE ---
            clean_name = clean_filename(audio_filename)
            if clean_name in processed_files:
                continue  # Skip this file, it's already done!
            
            # Check if file exists on disk
            audio_full_path = os.path.join(CLIPS_FOLDER, audio_filename)
            if not os.path.exists(audio_full_path):
                continue
            
            try:
                # Transcribe
                segments, info = model.transcribe(audio_full_path, language="he", beam_size=5)
                transcribed_text = " ".join([s.text for s in segments]).strip()
                
                # Cleanup strings
                clean_ref = reference_text.replace('\n', ' ').replace('\t', ' ').strip()
                clean_trans = transcribed_text.replace('\n', ' ').replace('\t', ' ').strip()
                
                # Write and Flush immediately
                f_out.write(f"{clean_name}\t{clean_ref}\t{clean_trans}\n")
                
            except Exception as e:
                print(f"Error processing {audio_filename}: {e}")

    print("\nDone! All files processed.")

if __name__ == "__main__":
    run_transcription_resumable()