import csv
from collections import Counter
import numpy as np

# --- 1. Alignment Helper Classes ---

class WordEditWeights:
    """
    Defines the cost for edit operations.
    Standard WER calculation uses:
    - Match = 0
    - Substitution / Insertion / Deletion = 1
    """
    def __init__(self):
        self.match_cost = 0
        self.substitution_cost = 1
        self.insertion_cost = 1
        self.deletion_cost = 1

def align_sequences(ref_tokens, hyp_tokens):
    """
    Implements the Needleman-Wunsch algorithm to align two lists of words.
    Returns a list of tuples: [(ref_word, hyp_word), ...]
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    weights = WordEditWeights()
    
    # Initialize Distance Matrix
    # dp[i][j] stores the minimum edit distance between ref[:i] and hyp[:j]
    dp = np.zeros((n + 1, m + 1))
    
    # Initialize first column (Deletions) and first row (Insertions)
    for i in range(n + 1):
        dp[i][0] = i * weights.deletion_cost
    for j in range(m + 1):
        dp[0][j] = j * weights.insertion_cost
        
    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_w = ref_tokens[i-1]
            hyp_w = hyp_tokens[j-1]
            
            # Calculate costs
            cost_match = dp[i-1][j-1] + (0 if ref_w == hyp_w else weights.substitution_cost)
            cost_del = dp[i-1][j] + weights.deletion_cost
            cost_ins = dp[i][j-1] + weights.insertion_cost
            
            dp[i][j] = min(cost_match, cost_del, cost_ins)
            
    # Backtracking to find the alignment path
    alignment = []
    i, j = n, m
    
    while i > 0 or j > 0:
        ref_w = ref_tokens[i-1] if i > 0 else None
        hyp_w = hyp_tokens[j-1] if j > 0 else None
        
        current_score = dp[i][j]
        
        # Determine which operation led to the current score
        # Priority: Match/Sub -> Delete -> Insert
        
        # Check for Match or Substitution
        if i > 0 and j > 0 and (current_score == dp[i-1][j-1] + (0 if ref_w == hyp_w else weights.substitution_cost)):
            alignment.append((ref_w, hyp_w))
            i -= 1
            j -= 1
        # Check for Deletion (Word in Ref missing in Hyp)
        elif i > 0 and (current_score == dp[i-1][j] + weights.deletion_cost):
            alignment.append((ref_w, "")) 
            i -= 1
        # Check for Insertion (Word in Hyp not in Ref)
        else:
            alignment.append(("", hyp_w))
            j -= 1
            
    return list(reversed(alignment))

# --- 2. AccuracyStatistics Class (The Core Logic) ---

class AccuracyStatistics:
    def __init__(self):
        # Counters
        self.n_gt = 0    # Total words in Reference
        self.n_asr = 0   # Total words in ASR Output
        self.hits = 0    # M (Matches)
        self.subs = 0    # S (Substitutions)
        self.ins = 0     # I (Insertions)
        self.dels = 0    # D (Deletions)
        
        # Error tracking for "Frequent Errors" analysis
        self.errors = Counter()

    def add_alignment(self, alignment):
        """
        Takes an alignment list [(ref, hyp), ...] and updates statistics.
        """
        for ref_w, hyp_w in alignment:
            # Update word counts
            if ref_w and ref_w != "": self.n_gt += 1
            if hyp_w and hyp_w != "": self.n_asr += 1
            
            # Classify the pair
            if ref_w == hyp_w:
                self.hits += 1 # Match
            elif ref_w != "" and hyp_w != "":
                self.subs += 1 # Substitution
                self.errors[(ref_w, hyp_w)] += 1
            elif ref_w != "" and hyp_w == "":
                self.dels += 1 # Deletion
                self.errors[(ref_w, "<Deleted>")] += 1
            elif ref_w == "" and hyp_w != "":
                self.ins += 1  # Insertion
                self.errors[("<Inserted>", hyp_w)] += 1

    def __iadd__(self, other):
        """
        Overloads the += operator to aggregate statistics from multiple files.
        """
        self.n_gt += other.n_gt
        self.n_asr += other.n_asr
        self.hits += other.hits
        self.subs += other.subs
        self.ins += other.ins
        self.dels += other.dels
        self.errors.update(other.errors)
        return self

    # --- Properties for Metrics ---
    
    @property
    def wer(self):
        # WER = (S + D + I) / N_gt
        if self.n_gt == 0: return 0.0
        return (self.subs + self.dels + self.ins) / self.n_gt

    @property
    def precision(self):
        # Precision = Matches / N_asr
        if self.n_asr == 0: return 0.0
        return self.hits / self.n_asr

    @property
    def recall(self):
        # Recall = Matches / N_gt
        if self.n_gt == 0: return 0.0
        return self.hits / self.n_gt

    @property
    def f1_score(self):
        # F1 = 2 * (Prec * Rec) / (Prec + Rec)
        p = self.precision
        r = self.recall
        if (p + r) == 0: return 0.0
        return 2 * (p * r) / (p + r)

    def frequent_errors(self, n=10):
        """Returns the top n most frequent errors."""
        return self.errors.most_common(n)

# --- 3. Main Processing Function ---

def process_results_part_b(input_file, output_file):
    print(f"Reading from: {input_file}")
    print(f"Writing to:   {output_file}")
    
    global_stats = AccuracyStatistics()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            
            # Read input file
            reader = csv.DictReader(f_in, delimiter='\t')
            
            # Prepare output file
            fieldnames = ['Filename', 'N_gt', 'N_asr', '#M', '#S', '#I', '#D', 
                          'WER', 'Recall', 'Precision', 'F1-Score']
            writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            count = 0
            for row in reader:
                # 1. Get Text
                filename = row['Filename']
                ref_text = row.get('Reference Text', '').strip()
                hyp_text = row.get('Transcribed Text', '').strip()
                
                # 2. Tokenize (Split by space)
                ref_tokens = ref_text.split()
                hyp_tokens = hyp_text.split()
                
                # 3. Align
                alignment = align_sequences(ref_tokens, hyp_tokens)
                
                # 4. Calculate Stats for this file
                file_stats = AccuracyStatistics()
                file_stats.add_alignment(alignment)
                
                # 5. Add to Global Stats
                global_stats += file_stats
                
                # 6. Write Row
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
            
            print(f"Processed {count} files.")

            # --- Write TOTAL Row ---
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
            
        print("\n=== Top 10 Frequent Errors ===")
        for (ref, hyp), freq in global_stats.frequent_errors(10):
            print(f'-> "{ref}" replaced by "{hyp}" : {freq} times.')
            
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Did you run Part A?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure this input filename matches what you generated in Part A
    INPUT_FILE = "results_part_a.tsv"
    OUTPUT_FILE = "results_part_b.tsv"
    
    process_results_part_b(INPUT_FILE, OUTPUT_FILE)