import csv
from collections import Counter
import numpy as np
from typing import Iterable, List, Tuple
import os

# ==========================================
# 1. YOUR PROVIDED CODE (Infrastructure)
# ==========================================

class EditWeights:
    """Abstract base class for edit weights."""
    def pair_weight(self, first_obj, second_obj) -> float: pass
    def insertion_weight(self, obj) -> float: pass
    def deletion_weight(self, obj) -> float: pass

def align_sequences(first_seq: Iterable, second_seq: Iterable, weights: EditWeights) -> Tuple[float, List[Tuple]]:
    """
    Your provided global alignment function.
    Returns: (score, aligned_pairs)
    """
    _OP_NULL = 0
    _OP_PAIR = 1
    _OP_INS = 2
    _OP_DEL = 3

    first_len = len(first_seq)
    second_len = len(second_seq)

    scores_mat = np.zeros((first_len + 1, second_len + 1))
    ops_mat = np.zeros((first_len + 1, second_len + 1), dtype=np.int8)

    scores_mat[0, 0] = 0
    ops_mat[0, 0] = _OP_NULL

    j = 1
    for second_obj in second_seq:
        ins_wgt = scores_mat[0, j - 1] + weights.insertion_weight(second_obj)
        scores_mat[0, j] = ins_wgt
        ops_mat[0, j] = _OP_INS
        j += 1

    i = 1
    for first_obj in first_seq:
        del_wgt = scores_mat[i - 1, 0] + weights.deletion_weight(first_obj)
        scores_mat[i, 0] = del_wgt
        ops_mat[i, 0] = _OP_DEL

        j = 1
        for second_obj in second_seq:
            max_wgt = scores_mat[i - 1, j - 1] + weights.pair_weight(first_obj, second_obj)
            best_op = _OP_PAIR

            ins_wgt = scores_mat[i, j - 1] + weights.insertion_weight(second_obj)
            if ins_wgt > max_wgt:
                max_wgt = ins_wgt
                best_op = _OP_INS

            del_wgt = scores_mat[i - 1, j] + weights.deletion_weight(first_obj)
            if del_wgt > max_wgt:
                max_wgt = del_wgt
                best_op = _OP_DEL

            scores_mat[i, j] = max_wgt
            ops_mat[i, j] = best_op
            j += 1
        i += 1

    aligned_pairs = []
    i = first_len
    j = second_len

    while i > 0 or j > 0:
        curr_op = ops_mat[i, j]
        if curr_op == _OP_PAIR:
            i -= 1; j -= 1
            aligned_pairs.append((first_seq[i], second_seq[j]))
        elif curr_op == _OP_INS:
            j -= 1
            aligned_pairs.append((None, second_seq[j]))
        elif curr_op == _OP_DEL:
            i -= 1
            aligned_pairs.append((first_seq[i], None))
        else:
            break

    aligned_pairs.reverse()
    return scores_mat[first_len, second_len], aligned_pairs


# ==========================================
# 2. IMPLEMENTATION FOR PART B
# ==========================================

class WordLevelEditWeights(EditWeights):
    """
    Defines weights for WER calculation.
    We treat it as 'Score Maximization':
    - Match: 0
    - Error (Sub/Ins/Del): -1
    Resulting Score = -(Number of Errors)
    """
    def pair_weight(self, a, b):
        return 0 if a == b else -1

    def insertion_weight(self, obj):
        return -1

    def deletion_weight(self, obj):
        return -1

class AccuracyStatistics:
    """
        Let us denote:
        NGT     - number of ground-truth words,
        NASR    - number of transcription words,
        M       - the number of matching words,
        S       - the number of substitutions,
        I       - the number of inserted words,
        D       - the number of deleted words.

    """
    def __init__(self):
        self.n_gt = 0    # Total words in Reference
        self.n_asr = 0   # Total words in ASR Output
        self.hits = 0    # M
        self.subs = 0    # S 
        self.ins = 0     # I
        self.dels = 0    # D 
        self.errors = Counter()

    def add_alignment(self, alignment):
        """
        Parses the alignment output from your function.
        Format is: (ref_word, hyp_word) where either can be None.
        """
        for ref_w, hyp_w in alignment:
            # Update Total Counts
            if ref_w is not None: self.n_gt += 1
            if hyp_w is not None: self.n_asr += 1
            
            # Classification Logic
            if ref_w is not None and hyp_w is not None:
                if ref_w == hyp_w:
                    self.hits += 1  # Match
                else:
                    self.subs += 1  # Substitution
                    self.errors[(ref_w, hyp_w)] += 1
            
            elif ref_w is not None and hyp_w is None:
                self.dels += 1      # Deletion
                self.errors[(ref_w, "<Deleted>")] += 1
                
            elif ref_w is None and hyp_w is not None:
                self.ins += 1       # Insertion
                self.errors[("<Inserted>", hyp_w)] += 1

    def __iadd__(self, other):
        """Allow summing stats: global_stats += file_stats"""
        self.n_gt += other.n_gt
        self.n_asr += other.n_asr
        self.hits += other.hits
        self.subs += other.subs
        self.ins += other.ins
        self.dels += other.dels
        self.errors.update(other.errors)
        return self

    # --- Metrics Properties ---
    @property
    def wer(self):
        if self.n_gt == 0: return 0.0
        return (self.subs + self.dels + self.ins) / self.n_gt

    @property
    def precision(self):
        if self.n_asr == 0: return 0.0
        return self.hits / self.n_asr

    @property
    def recall(self):
        if self.n_gt == 0: return 0.0
        return self.hits / self.n_gt

    @property
    def f1_score(self):
        p = self.precision
        r = self.recall
        if (p + r) == 0: return 0.0
        return 2 * (p * r) / (p + r)

    def frequent_errors(self, n=10):
        return self.errors.most_common(n)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def process_results_part_b(input_file, output_file):
    print(f"Reading from: {input_file}")
    
    # Initialize our weights class
    weights = WordLevelEditWeights()
    global_stats = AccuracyStatistics()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            
            reader = csv.DictReader(f_in, delimiter='\t')
            
            # Define Output Columns
            fieldnames = ['Filename', 'N_gt', 'N_asr', '#M', '#S', '#I', '#D', 
                          'WER', 'Recall', 'Precision', 'F1-Score']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            count = 0
            for row in reader:
                filename = row['Filename']
                ref_text = row.get('Reference Text', '').strip()
                hyp_text = row.get('Transcribed Text', '').strip()
                
                # Tokenize (Split by space)
                ref_tokens = ref_text.split()
                hyp_tokens = hyp_text.split()
                
                # --- USE YOUR ALIGNMENT FUNCTION ---
                # We ignore the score ([0]) and take the alignment list ([1])
                _, alignment = align_sequences(ref_tokens, hyp_tokens, weights)
                
                # Calculate Stats
                file_stats = AccuracyStatistics()
                file_stats.add_alignment(alignment)
                
                # Add to Global
                global_stats += file_stats
                
                # Write Row
                writer.writerow({
                    'Filename': filename,
                    'N_gt': file_stats.n_gt,
                    'N_asr': file_stats.n_asr,
                    '#M': file_stats.hits,
                    '#S': file_stats.subs,
                    '#I': file_stats.ins,
                    '#D': file_stats.dels,
                    'WER': f"{file_stats.wer:.4f}",
                    'Recall': f"{file_stats.recall:.4f}",
                    'Precision': f"{file_stats.precision:.4f}",
                    'F1-Score': f"{file_stats.f1_score:.4f}"
                })
                count += 1
            
            # Write TOTAL Row
            writer.writerow({
                'Filename': 'TOTAL',
                'N_gt': global_stats.n_gt,
                'N_asr': global_stats.n_asr,
                '#M': global_stats.hits,
                '#S': global_stats.subs,
                '#I': global_stats.ins,
                '#D': global_stats.dels,
                'WER': f"{global_stats.wer:.4f}",
                'Recall': f"{global_stats.recall:.4f}",
                'Precision': f"{global_stats.precision:.4f}",
                'F1-Score': f"{global_stats.f1_score:.4f}"
            })
            
        print(f"Done! Processed {count} files.")
        print(f"Results written to {output_file}")

        print("\n=== Top 10 Frequent Errors ===")
        for (ref, hyp), freq in global_stats.frequent_errors(10):
            print(f'-> "{ref}" replaced by "{hyp}" : {freq} times.')
            
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please run Part A first.")

if __name__ == "__main__":
    INPUT_FILE = "results_part_a.tsv"
    OUTPUT_FILE = "results_part_b.tsv"
    process_results_part_b(INPUT_FILE, OUTPUT_FILE)