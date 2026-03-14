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
    if not text:
        return ""

    # 2. STANDARDIZE DASHES FIRST (Prioritize over deletion)
    # Replace all hyphen/dash types with a SPACE to keep words separate
    for char in ['-', '–', '—', '_', '־', '’']: # check where ’ is better
        text = text.replace(char, ' ')

    # 1. Remove Hebrew Niqqud
    text = re.sub(r'[\u0591-\u05C7]', '', text)

    # 3. Handle Hebrew-specific marks
    # Gershayim (״) usually joins acronyms (צה״ל -> צהל), so we use empty string here
    text = text.replace('״', '')

    # 4. Remove Other Punctuation
    # Use empty string for marks that don't separate words (like periods at end of sentences)
    for char in '.,?!:;"\'()[]{}<>“‘’“”':
        text = text.replace(char, '')


    # 4. Manual Fixes
    replacements = {
        "היתה": "הייתה",
        "הכול": "הכל",
        "מייד": "מיד",
        "ואלו": "ואילו",
        "איתי": "אתי",
        "מסים": "מיסים",
        "קוקטיל": "קוקטייל",
        "הבין לאומיים": "הבינלאומיים",
        "בית תפילה": "בית תפילה",
        "והאמנות": "והאומנות",
        "מאד": "מאוד",
        "אליי": "אלי",
        "ייפלו": "יפלו",
        "בספוטיפיי": "בספוטיפי",
        "ששייך": "ששיך",
        "באזניך": "באוזניך",
        "לעסת": "לעיסת",
        "בידים": "בידיים",
        "אשה": "אישה",
        "אלהיהם": "אלוהיהם",
        "ומתכוצים": "ומתכווצים",
        "כלם": "כולם",
        "לעתים": "לעיתים",
        "גלינו": "גילינו",
        "כשבועים": "כשבועיים",
        "במדה": "במידה",
        "אימה": "איומה",
        "מלכדת": "מלכודת",
        "אפלו": "אפילו",
        "מטתה": "מיטתה",
        "באופל": "באפל",
        "עכשו": "עכשיו",
        "תחזר": "תחזור",
        "לדוגמה": "לדוגמא",
        "גיהינום": "גיהנום",
        "מינהלי": "מנהלי",
        "גזירות": "גזרות",
        "ליצג": "לייצג",
        "פיסבוק": "פייסבוק",
        "אלטרנטיבים": "אלטרנטיביים",
        "הריינו": "הרינו",
        "לעיפה": "לעייפה",
        "כישרונות": "כשרונות",
        "הזיקנה": "הזקנה",
        "אהרן": "אהרון",
        "בדברי": "בדבריי",
        "המליונים": "המיליונים",
        "בהעדר": "בהיעדר",
        "התישבות": "התיישבות",
        "היעדר": "העדר",
        "ליבי": "לבי",
        "מצדם": "מצידם",
        "צפורה": "ציפורה",
        "תסע": "תיסע",
        "זיכרונות": "זכרונות",
        "נהרייה": "נהריה",
        "יקח ": "ייקח ",
        "לקסקלי": "לקסיקלי",
        "ומכער": "ומכוער",
        "המינהליות": "המנהליות",
        "לאפנו": "לאפינו",
        " רבותי": " רבותיי",
        "המלים": "המילים",
        "אינתיפדת": "אינתיפאדת",
        "שתים": "שתיים",

        " דר ": " דוקטור ",

        " ב ": "ב ",
        " ל ": "ל ",
        " מ ": "מ ",
        " כ ": "כ ",

        "בשעה ארבע": "בשעה 16",
        "שבעים אחוזים": "70%",
        "חמישה אחוז": "5%",
        "שמונים אחוזים": "80%",
        "עשרים ותשעה אחוזים": "29%",
        "עשרים ושישה": "26",
        "ארבע עשרה": "14",
        "כארבע מאות אלף": "400000כ",
        "מאתיים שמונים ואחד אלף": "281000",
        "מאה וחמישים אלף": "150000",
        "שלושים אלף": "30000",
        "עשרים ושלושה אלף": "23000",
        "עשרים ושניים אלף": "22000",
        "חמשת אלפים": "5000",
        "אלפיים ואחת עשרה": "2011",
        "אלפיים ושלוש עשרה": "2013",
        "אלפיים ושתים עשרה": "2012",
        "אלפיים ושמונה": "2008",
        "אלפיים ושש": "2006",
        "אלפיים ושלוש": "2003",
        "אלפיים": "2000",
        "אלף מאה ושמונים": "1180",
        "אלף וחמש מאות": "1500",
        "אלף תשע מאות שמונים ושמונה": "1988",
        "אלף תשע מאות חמישים ושש": "1956",
        "ארבע מאות": "400",
        "שלוש מאות" : "300",
        "מאה וארבעים": "140",
        "עשרים ותשעה": "29",
        "עשרים": "20",
        "שתים עשרה": "12",
        "חמישה עשר": "15",
        "שלוש עשרה": "13",
        "שבעה": "7",
        "שבע": "7",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def process_results_part_c(input_file, output_file):
    print(f"Reading from: {input_file}")
    weights = WordLevelEditWeights()
    global_stats = AccuracyStatistics()
    
    file_count = 0
    sum_file_wer = 0.0  # Track sum of individual file WERs for macro-average
    error_examples = {}  # Track example sentences for each error (ref, hyp) -> (raw_ref, raw_hyp)
    
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
                
                # Track error examples (for printing later)
                for (ref_w, hyp_w), count in file_stats.errors.items():
                    key = (ref_w, hyp_w)
                    if key not in error_examples:
                        error_examples[key] = (raw_ref, raw_hyp)
                
                # Track for macro-average calculation
                file_count += 1
                sum_file_wer += file_stats.wer
                
                # Write simple row
                writer.writerow({
                    'Filename': filename,
                    'N_gt': file_stats.n_gt, 'N_asr': file_stats.n_asr,
                    '#M': file_stats.hits, '#S': file_stats.subs,
                    '#I': file_stats.ins, '#D': file_stats.dels,
                    'WER': f"{file_stats.wer:.4f}"
                })

            if file_count > 0:
                # ROW 1: TOTAL (SUM) - Micro-average: total errors / total words
                writer.writerow({
                    'Filename': 'TOTAL (SUM)',
                    'N_gt': global_stats.n_gt, 'N_asr': global_stats.n_asr,
                    '#M': global_stats.hits, '#S': global_stats.subs,
                    '#I': global_stats.ins, '#D': global_stats.dels,
                    'WER': f"{global_stats.wer:.4f}"
                })
                
                # ROW 2: AVERAGE - Macro-average: average of individual file WERs
                writer.writerow({
                    'Filename': 'AVERAGE',
                    'N_gt': f"{global_stats.n_gt / file_count:.2f}",
                    'N_asr': f"{global_stats.n_asr / file_count:.2f}",
                    '#M': f"{global_stats.hits / file_count:.2f}",
                    '#S': f"{global_stats.subs / file_count:.2f}",
                    '#I': f"{global_stats.ins / file_count:.2f}",
                    '#D': f"{global_stats.dels / file_count:.2f}",
                    'WER': f"{sum_file_wer / file_count:.4f}"
                })
                
                # ROW 3: SUM (FILE WER) - Sum of all individual file WERs
                writer.writerow({
                    'Filename': 'SUM (FILE WER)',
                    'N_gt': '', 'N_asr': '',
                    '#M': '', '#S': '', '#I': '', '#D': '',
                    'WER': f"{sum_file_wer:.4f}"
                })

        print(f"Done! Processed {file_count} files. Results written to {output_file}")
        
        print("\n=== FINAL CORPUS METRICS ===")
        print(f"Total WER (Micro-Average): {global_stats.wer:.4f}")
        if file_count > 0:
            print(f"Average WER (Macro-Average): {sum_file_wer / file_count:.4f}")
            print(f"Sum of file WERs: {sum_file_wer:.4f}")

        print("\n=== Top 300 Frequent Errors (After Normalization) ===")
        print("Use this list to improve your normalize_text function!")
        for (ref, hyp), freq in global_stats.frequent_errors(200):
            print(f'-> "{ref}" replaced by "{hyp}" : {freq} times.')
            # Show example sentences
            if (ref, hyp) in error_examples:
                raw_ref, raw_hyp = error_examples[(ref, hyp)]
                print(f'{(raw_ref)} | {(raw_hyp)}')
                print(f'{normalize_text(raw_ref)} | {normalize_text(raw_hyp)}')
                # print(f'   Original normalized: {normalize_text(raw_ref)}')
                # print(f'   Transcribed normalized: {normalize_text(raw_hyp)}')

    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Run Part A first.")

if __name__ == "__main__":
    process_results_part_c("results_part_a_noisy_clips.tsv", "results_part_c_noisy_clips.tsv")