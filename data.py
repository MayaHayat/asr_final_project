import os

# --- Update this path to your folder ---
DATASET_PATH = r"C:\Users\Maya\OneDrive\Desktop\Masters\asr_final_proj\cv-corpus-24.0-2025-12-05-he.tar\cv-corpus-24.0-2025-12-05\he"
TEST_TSV_FILE = os.path.join(DATASET_PATH, "test.tsv")

def explore_tsv():
    print(f"--- Exploring: {TEST_TSV_FILE} ---")
    
    try:
        with open(TEST_TSV_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 1. Basic Stats
        total_lines = len(lines)
        print(f"Total Lines: {total_lines}")
        print(f"Total Samples (minus header): {total_lines - 1}")
        print("-" * 50)

        # 2. The Header (Column Names)
        header = lines[0].strip().split('\t')
        print("COLUMNS FOUND:")
        for idx, col in enumerate(header):
            print(f"  [{idx}] {col}")
        print("-" * 50)

        # 3. First 5 Samples
        print("FIRST 5 SAMPLES:")
        for i in range(1, min(6, total_lines)):
            row = lines[i].strip().split('\t')
            print(f"\nSample #{i}:")
            # Print only the important columns to keep it readable
            # We assume index 1 is 'path' and index 2 is 'sentence' based on standard Common Voice
            if len(row) > 2:
                for j in range(len(row)):
                    print(f"  Column {j}: {row[j]}")
                # print(f"  Filename: {row[1]}") 
                # print(f"  Text:     {row[2]}")
                # # Optional: Print metadata if it exists
                # if len(row) > 6:
                #      print(f"  Gender:   {row[6]}") 
                #      print(f"  Age:      {row[5]}")

    except FileNotFoundError:
        print("Error: File not found. Check your path!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    explore_tsv()