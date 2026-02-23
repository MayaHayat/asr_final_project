import csv
import re
from collections import Counter
import numpy as np
from typing import Iterable, List, Tuple

# ==========================================
# 1. ALIGNMENT INFRASTRUCTURE (Same as Part B)
# ==========================================

class EditWeights:
    def pair_weight(self, first_obj, second_obj) -> float: pass
    def insertion_weight(self, obj) -> float: pass
    def deletion_weight(self, obj) -> float: pass

def align_sequences(first_seq: Iterable, second_seq: Iterable, weights: EditWeights) -> Tuple[float, List[Tuple]]:
    """ Standard Needleman-Wunsch Alignment """
    _OP_NULL, _OP_PAIR, _OP_INS, _OP_DEL = 0, 1, 2, 3
    first_len, second_len = len(first_seq), len(second_seq)
    scores_mat = np.zeros((first_len + 1, second_len + 1))
    ops_mat = np.zeros((first_len + 1, second_len + 1), dtype=np.int8)

    # Init
    scores_mat[0, 0] = 0
    ops_mat[0, 0] = _OP_NULL
    
    j = 1
    for second_obj in second_seq:
        scores_mat[0, j] = scores_mat[0, j - 1] + weights.insertion_weight(second_obj)
        ops_mat[0, j] = _OP_INS
        j += 1
    i = 1
    for first_obj in first_seq:
        scores_mat[i, 0] = scores_mat[i - 1, 0] + weights.deletion_weight(first_obj)
        ops_mat[i, 0] = _OP_DEL
        j = 1
        for second_obj in second_seq:
            pair_val = scores_mat[i - 1, j - 1] + weights.pair_weight(first_obj, second_obj)
            ins_val = scores_mat[i, j - 1] + weights.insertion_weight(second_obj)
            del_val = scores_mat[i - 1, j] + weights.deletion_weight(first_obj)
            
            best_val = pair_val
            best_op = _OP_PAIR
            
            if ins_val > best_val:
                best_val = ins_val
                best_op = _OP_INS
            if del_val > best_val:
                best_val = del_val
                best_op = _OP_DEL
                
            scores_mat[i, j] = best_val
            ops_mat[i, j] = best_op
            j += 1
        i += 1

    # Backtrack
    aligned_pairs = []
    i, j = first_len, second_len
    while i > 0 or j > 0:
        op = ops_mat[i, j]
        if op == _OP_PAIR:
            i -= 1; j -= 1
            aligned_pairs.append((first_seq[i], second_seq[j]))
        elif op == _OP_INS:
            j -= 1
            aligned_pairs.append((None, second_seq[j]))
        elif op == _OP_DEL:
            i -= 1
            aligned_pairs.append((first_seq[i], None))
        else: break
    aligned_pairs.reverse()
    return scores_mat[first_len, second_len], aligned_pairs

class WordLevelEditWeights(EditWeights):
    def pair_weight(self, a, b): return 0 if a == b else -1
    def insertion_weight(self, obj): return -1
    def deletion_weight(self, obj): return -1

class AccuracyStatistics:
    def __init__(self):
        self.n_gt = 0
        self.n_asr = 0
        self.hits = 0
        self.subs = 0
        self.ins = 0
        self.dels = 0
        self.errors = Counter()

    def add_alignment(self, alignment):
        for ref_w, hyp_w in alignment:
            if ref_w is not None: self.n_gt += 1
            if hyp_w is not None: self.n_asr += 1
            
            if ref_w is not None and hyp_w is not None:
                if ref_w == hyp_w: self.hits += 1
                else: 
                    self.subs += 1
                    self.errors[(ref_w, hyp_w)] += 1
            elif ref_w is not None and hyp_w is None:
                self.dels += 1
                self.errors[(ref_w, "<Deleted>")] += 1
            elif ref_w is None and hyp_w is not None:
                self.ins += 1
                self.errors[("<Inserted>", hyp_w)] += 1

    def __iadd__(self, other):
        self.n_gt += other.n_gt
        self.n_asr += other.n_asr
        self.hits += other.hits
        self.subs += other.subs
        self.ins += other.ins
        self.dels += other.dels
        self.errors.update(other.errors)
        return self

    @property
    def wer(self): return (self.subs + self.dels + self.ins) / self.n_gt if self.n_gt > 0 else 0.0
    
    def frequent_errors(self, n=20): return self.errors.most_common(n)

# ==========================================
# 2. NORMALIZATION LOGIC (THE NEW PART)
# ==========================================

def normalize_text(text):
    """
    Cleans text to ensure fair comparison.
    """
    if not text:
        return ""

    # 1. Remove Hebrew Niqqud (Vowels) [Range 0591-05C7]
    # Common Voice sometimes has vowels, Whisper usually doesn't.
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # 2. Remove Punctuation
    # Whisper adds periods and commas. We should remove them for scoring.
    # We keep only letters (Hebrew/English) and spaces.
    for char in '.,?!:;"\'()[]{}<>':
        text = text.replace(char, '')
    
    # 3. Replace all types of dashes with a space
    for char in ['-', '–', '—', '_']:
        text = text.replace(char, ' ')

    # 4. Manual Fixes
    text = text.replace("היתה", "הייתה")
    

    return text

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def process_results_part_c(input_file, output_file):
    print(f"Reading from: {input_file}")
    weights = WordLevelEditWeights()
    global_stats = AccuracyStatistics()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            
            reader = csv.DictReader(f_in, delimiter='\t')
            fieldnames = ['Filename', 'N_gt', 'N_asr', '#M', '#S', '#I', '#D', 'WER']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for row in reader:
                filename = row['Filename']
                
                # --- APPLY NORMALIZATION HERE ---
                raw_ref = row.get('Reference Text', '')
                raw_hyp = row.get('Transcribed Text', '')
                
                ref_text = normalize_text(raw_ref)
                hyp_text = normalize_text(raw_hyp)
                
                # Align Normalized Text
                _, alignment = align_sequences(ref_text.split(), hyp_text.split(), weights)
                
                # Stats
                file_stats = AccuracyStatistics()
                file_stats.add_alignment(alignment)
                global_stats += file_stats
                
                # Write simple row
                writer.writerow({
                    'Filename': filename,
                    'N_gt': file_stats.n_gt, 'N_asr': file_stats.n_asr,
                    '#M': file_stats.hits, '#S': file_stats.subs,
                    '#I': file_stats.ins, '#D': file_stats.dels,
                    'WER': f"{file_stats.wer:.4f}"
                })

            # TOTAL ROW
            writer.writerow({'Filename': 'TOTAL', 
                             'N_gt': global_stats.n_gt, 'N_asr': global_stats.n_asr,
                             '#M': global_stats.hits, '#S': global_stats.subs,
                             '#I': global_stats.ins, '#D': global_stats.dels,
                             'WER': f"{global_stats.wer:.4f}"})

        print(f"Done! Results written to {output_file}")
        
        print("\n=== FINAL TOTAL WER ===")
        print(f"WER: {global_stats.wer:.4f}")

        print("\n=== Top 20 Frequent Errors (After Normalization) ===")
        print("Use this list to improve your normalize_text function!")
        for (ref, hyp), freq in global_stats.frequent_errors(20):
            print(f'-> "{ref}" replaced by "{hyp}" : {freq} times.')

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Run Part A first.")

if __name__ == "__main__":
    process_results_part_c("results_part_a.tsv", "results_part_c.tsv")