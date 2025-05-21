import re
import sys

try:
    from word2number import w2n
except ImportError:
    print("Warning: 'word2number' library not found. Number parsing from words will be limited. "
          "Please install it using: pip install word2number")
    w2n = None

nltk_module = None
lemmatizer = None
word_tokenize_func = None
all_nltk_resources_available = False

try:
    import nltk
    nltk_module = nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    word_tokenize_func = word_tokenize

    nltk_resources_to_check = {
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'punkt': 'tokenizers/punkt'
    }

    successfully_loaded_resources_count = 0

    for resource_name, resource_path_fragment in nltk_resources_to_check.items():
        try:
            nltk_module.data.find(resource_path_fragment)
            successfully_loaded_resources_count += 1
        except LookupError:
            try:
                nltk_module.download(resource_name, quiet=True)
                nltk_module.data.find(resource_path_fragment)
                successfully_loaded_resources_count += 1
            except Exception as e_download:
                print(f"ERROR (check.py): Failed to download/verify NLTK resource '{resource_name}': {e_download}")

    if successfully_loaded_resources_count == len(nltk_resources_to_check):
        all_nltk_resources_available = True
        lemmatizer = WordNetLemmatizer()
    else:
        lemmatizer = None

except ImportError:
    print("NLTK Warning (check.py): 'nltk' library not found. Lemmatization disabled.")

CONTRACTIONS_VQA = {
    'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've",
    'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't",
    'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's",
    "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've",
    'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've",
    'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't",
    'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll",
    'somebodys': "somebody's", 'someoned': "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    'someonell': "someone'll", 'someones': "someone's", 'somethingd': "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've",
    'somethingll': "something'll", 'thats': "that's", 'thered': "there'd",
    "thered've": "there'd've", "there'dve": "there'd've", 'therere': "there're",
    'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're",
    'theyve': "they've", 'twas': "'twas", 'wasnt': "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", 'weve': "we've",
    'werent': "weren't", 'whatll': "what'll", 'whatre': "what're",
    'whats': "what's", 'whatve': "what've", 'whens': "when's",
    'whered': "where'd", 'wheres': "where's", 'whereve': "where've",
    'whod': "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    'wholl': "who'll", 'whos': "who's", 'whove': "who've", 'whyll': "why'll",
    'whyre': "why're", 'whys': "why's", 'wont': "won't", 'wouldve': "would've",
    'wouldnt': "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", 'youd': "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll",
    'youre': "you're", 'youve': "you've",
}
MANUAL_MAP_VQA = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10',
}
ARTICLES_VQA = {'a', 'an', 'the'}
PUNCT_VQA = [';', '/', '[', ']', '"', '{', '}', '(', ')',
             '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
COMMA_RE_VQA = re.compile(r'(\d)(,)(\d)')
PERIOD_RE_VQA = re.compile(r'(?!<=\d)(\.)(?!\d)')

def normalize_vqa(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = COMMA_RE_VQA.sub(r'\1\3', text)
    for p in PUNCT_VQA:
        text = text.replace(p, ' ')
    text = PERIOD_RE_VQA.sub('', text)
    words = []
    for w in text.split():
        w = MANUAL_MAP_VQA.get(w, w)
        if w in ARTICLES_VQA:
            continue
        words.append(CONTRACTIONS_VQA.get(w, w))
    return ' '.join(words)

def clean_answer_for_comparison(answer_text: str) -> str:
    if not isinstance(answer_text, str):
        return ""

    text_to_clean = answer_text.strip()
    match = re.match(r"^(?:\[Answer\]|\[Reattempted Answer\])\s*(.*)", text_to_clean, re.IGNORECASE)
    if match:
        text_to_clean = match.group(1)

    return text_to_clean.strip().lower()

def parse_number(text: str):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        if w2n:
            try:
                return float(w2n.word_to_num(text))
            except ValueError:
                pass

        numbers_found = re.findall(r'-?\d+\.?\d*|-?\.\d+', text)
        if numbers_found:
            try:
                return float(numbers_found[0])
            except ValueError:
                return None
        return None

def get_lemmatized_tokens(text: str) -> list:
    if not all_nltk_resources_available or not lemmatizer or not word_tokenize_func:
        return [t for t in re.findall(r'\b\w+\b', text.lower()) if t] if text else []

    if not isinstance(text, str) or not text.strip():
        return []

    try:
        tokens = word_tokenize_func(text.lower())
        processed_tokens = []
        for token in tokens:
            lemma = None
            if token == 'kiting':
                lemma = 'kite'
            elif token.endswith('ing'):
                lemma = lemmatizer.lemmatize(token, pos='v')
            elif token == 'wooden':
                lemma = 'wood'
            elif token == 'metallic':
                lemma = 'metal'
            else:
                lemma = lemmatizer.lemmatize(token, pos='n')

            if lemma:
                processed_tokens.append(lemma)
        
        final_tokens = [t for t in processed_tokens if t]
        return final_tokens
    except Exception as e:
        return [t for t in re.findall(r'\b\w+\b', text.lower()) if t] if text else []


def calculate_token_f1(model_ans_str: str, target_ans_str: str, use_lemmatization: bool = True) -> float:
    if use_lemmatization and all_nltk_resources_available:
        processed_model_tokens = get_lemmatized_tokens(model_ans_str if model_ans_str else "")
        processed_target_tokens = get_lemmatized_tokens(target_ans_str if target_ans_str else "")
    else:
        processed_model_tokens = [t for t in (model_ans_str.lower().split() if model_ans_str else []) if t]
        processed_target_tokens = [t for t in (target_ans_str.lower().split() if target_ans_str else []) if t]

    if not processed_model_tokens and not processed_target_tokens:
        return 1.0
    if not processed_model_tokens or not processed_target_tokens:
        return 0.0

    common_tokens = set(processed_model_tokens) & set(processed_target_tokens)

    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(processed_model_tokens)
    recall = len(common_tokens) / len(processed_target_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def perform_direct_accuracy_check(
    final_answer_text: str,
    target_answers: any,
    question_type: str,
    current_error: str = None,
    verbose: bool = False,
    f1_threshold_other: float = 0.5,
    enable_relaxed_other_check: bool = True
) -> tuple:
    direct_accuracy_results = {
        "is_correct": None,
        "cleaned_model_answer": "",
        "normalized_target_answer": "",
        "normalized_target_answers": [],
        "vqa_score": None,
        "notes": ""
    }
    grades_output = ["Direct Comparison: Not Performed"]
    majority_vote_output = "Direct comparison: Not Performed"
    match_method_note = "No check performed"

    if current_error:
        direct_accuracy_results["notes"] = "Skipped due to pipeline error."
        grades_output = ["Direct Comparison Skipped - Pipeline Error"]
        majority_vote_output = "Direct comparison: Skipped due to pipeline error"
        return direct_accuracy_results, grades_output, majority_vote_output

    if isinstance(target_answers, str):
        processed_target_answer_list = [str(target_answers).strip()]
    elif isinstance(target_answers, list):
        processed_target_answer_list = [str(t).strip() for t in target_answers if t is not None]
    else:
        processed_target_answer_list = []
        
    if not processed_target_answer_list:
        direct_accuracy_results["notes"] = "Skipped due to missing or empty target_answer(s)."
        grades_output = ["Direct Comparison Skipped - No Target Answer(s)"]
        majority_vote_output = "Direct comparison: Skipped - No Target Answer(s)"
        return direct_accuracy_results, grades_output, majority_vote_output

    processed_target_answer_for_single_comparison = processed_target_answer_list[0]
    cleaned_model_ans = clean_answer_for_comparison(final_answer_text if final_answer_text else "")
    normalized_target_ans_single = processed_target_answer_for_single_comparison.lower() 
    is_correct = False

    direct_accuracy_results["cleaned_model_answer"] = cleaned_model_ans
    direct_accuracy_results["normalized_target_answer"] = normalized_target_ans_single
    direct_accuracy_results["normalized_target_answers"] = [t.lower() for t in processed_target_answer_list]

    if not cleaned_model_ans and cleaned_model_ans != "0": # "0" can be a valid answer
        is_correct = (normalized_target_ans_single == "" or normalized_target_ans_single == "none")
        match_method_note = "Model answer empty after cleaning."
        if question_type == "vqa_standard":
            vqa_model_ans_norm = normalize_vqa(cleaned_model_ans)
            matches = 0
            for gt_ans in processed_target_answer_list:
                gt_ans_norm = normalize_vqa(gt_ans)
                if vqa_model_ans_norm == gt_ans_norm:
                    matches += 1
            vqa_s = min(matches / 3.0, 1.0)
            direct_accuracy_results["vqa_score"] = vqa_s
            is_correct = vqa_s > 0
            match_method_note = f"VQA Standard: Model empty. Score: {vqa_s:.2f}."

    elif question_type == "vqa_standard":
        if not isinstance(target_answers, list) or not target_answers:
            match_method_note = "VQA Standard: target_answers must be a non-empty list."
            is_correct = False
        else:
            model_ans_vqa_normalized = normalize_vqa(cleaned_model_ans)
            gt_ans_vqa_normalized_list = [normalize_vqa(gt) for gt in processed_target_answer_list]
            direct_accuracy_results["normalized_target_answers"] = gt_ans_vqa_normalized_list

            matches = 0
            for gt_norm in gt_ans_vqa_normalized_list:
                if model_ans_vqa_normalized == gt_norm:
                    matches += 1
            
            vqa_s = min(matches / 3.0, 1.0)
            direct_accuracy_results["vqa_score"] = vqa_s
            is_correct = (vqa_s > 0) 
            match_method_note = f"VQA Standard: Score {vqa_s:.2f} ({matches} matches)."

    elif question_type == "yes/no":
        positive_indicators = {"yes", "yeah", "yep", "correct", "true", "affirmative", "y", "positive"}
        negative_indicators = {"no", "nope", "incorrect", "false", "negative", "n", "not"}

        model_ans_is_positive = cleaned_model_ans in positive_indicators
        model_ans_is_negative = cleaned_model_ans in negative_indicators
        target_is_positive = normalized_target_ans_single in positive_indicators
        target_is_negative = normalized_target_ans_single in negative_indicators

        if (model_ans_is_positive and target_is_positive) or \
           (model_ans_is_negative and target_is_negative):
            is_correct = True
            match_method_note = "Yes/No variants check (standard)."
        elif not (target_is_positive or target_is_negative) and not (model_ans_is_positive or model_ans_is_negative):
            if cleaned_model_ans == normalized_target_ans_single:
                is_correct = True
                match_method_note = "Yes/No variants check (exact match on non-standard words)."
            else:
                f1_yes_no_fallback = calculate_token_f1(cleaned_model_ans, normalized_target_ans_single, use_lemmatization=all_nltk_resources_available)
                if f1_yes_no_fallback >= 0.5:
                    is_correct = True
                    match_method_note = f"Yes/No variants check (F1 match on non-standard: {f1_yes_no_fallback:.2f})."
                else:
                    match_method_note = f"Yes/No variants check (non-standard words no match, F1={f1_yes_no_fallback:.2f})."
        else:
            match_method_note = "Yes/No variants check (mismatch)."
            if not (target_is_positive or target_is_negative) and cleaned_model_ans == normalized_target_ans_single:
                 is_correct = True
                 match_method_note = "Yes/No variants check (model matched specific non-yes/no target)."

    elif question_type == "number":
        model_num = parse_number(cleaned_model_ans)
        target_num = parse_number(normalized_target_ans_single)

        if model_num is not None and target_num is not None:
            if isinstance(model_num, float) or isinstance(target_num, float):
                is_correct = abs(model_num - target_num) < 1e-5 # Tolerance for float comparison
            else:
                is_correct = int(model_num) == int(target_num)
            match_method_note = f"Number parsed (model={model_num}, target={target_num})."
        else:
            is_correct = (cleaned_model_ans == normalized_target_ans_single) # Fallback
            match_method_note = f"Number string match (parsed: model={model_num}, target={target_num})."
            
    elif question_type == "other":
        if cleaned_model_ans == normalized_target_ans_single:
            is_correct = True
            match_method_note = "Other: Exact match."
        else:
            can_use_lemma = all_nltk_resources_available
            f1 = calculate_token_f1(cleaned_model_ans, normalized_target_ans_single, use_lemmatization=can_use_lemma)

            if f1 >= f1_threshold_other:
                is_correct = True
                match_method_note = f"Other: Token F1 {'(lemmatized)' if can_use_lemma else '(no lemma)'} ({f1:.2f} >= {f1_threshold_other})."
            elif enable_relaxed_other_check and can_use_lemma:
                original_f1_fail_note_prefix = f"Other: Token F1 (lemmatized) ({f1:.2f} < {f1_threshold_other})."
                l_model_tokens = get_lemmatized_tokens(cleaned_model_ans)
                l_target_tokens = get_lemmatized_tokens(normalized_target_ans_single)
                relaxed_match_found = False

                if l_model_tokens and l_target_tokens:
                    set_model = set(l_model_tokens)
                    set_target = set(l_target_tokens)
                    len_model = len(l_model_tokens)
                    len_target = len(l_target_tokens)

                    if (set_model.issubset(set_target) or set_target.issubset(set_model)) and \
                       abs(len_model - len_target) <= 2 and (set_model & set_target):
                        # More lenient for very short answers
                        if (len_model <= 2 and len_target <= 3) or \
                           (len_target <= 2 and len_model <= 3) or \
                           (len_model ==1 and len_target > 0) or \
                           (len_target ==1 and len_model > 0) :
                            is_correct = True
                            relaxed_match_found = True
                            match_method_note = f"{original_f1_fail_note_prefix} Relaxed: Token subset match."
                    
                    if not is_correct and len_model == 1 and len_target == 1:
                        m_tok_lemma = l_model_tokens[0]
                        t_tok_lemma = l_target_tokens[0]
                        if len(m_tok_lemma) >= 3 and len(t_tok_lemma) >=3: # Avoid tiny words
                            common_suffixes = ['s', 'es', 'ing', 'ed', 'er', 'est', 'en', 'al', 'ic', 'ive', 'ous', 'ly', 'tion', 'sion', 'ment']
                            if (m_tok_lemma.startswith(t_tok_lemma) and m_tok_lemma[len(t_tok_lemma):] in common_suffixes) or \
                               (t_tok_lemma.startswith(m_tok_lemma) and t_tok_lemma[len(m_tok_lemma):] in common_suffixes):
                                is_correct = True
                                relaxed_match_found = True
                                match_method_note = f"{original_f1_fail_note_prefix} Relaxed: Morphological variant (prefix/suffix) for '{m_tok_lemma}' vs '{t_tok_lemma}'."

                if not relaxed_match_found:
                    match_method_note = f"{original_f1_fail_note_prefix} Relaxed checks failed or not applicable."
                elif not is_correct :
                     match_method_note = f"{original_f1_fail_note_prefix} Relaxed checks logic error."
            else:
                match_method_note = f"Other: No match (Exact & F1 {'(lemmatized)' if can_use_lemma else '(no lemma)'} < {f1_threshold_other} failed, F1={f1:.2f}). Relaxed checks disabled or NLTK unavailable."
    else:
        is_correct = (cleaned_model_ans == normalized_target_ans_single)
        match_method_note = f"Unknown QType ({question_type}): Exact match with first target."

    direct_accuracy_results["is_correct"] = is_correct
    direct_accuracy_results["notes"] = match_method_note
    
    if verbose:
        print_color_prefix = "\033[94m"
        print_color_suffix = "\033[0m"
        try:
            if not sys.stdout.isatty(): # Check if output is not a TTY
                 print_color_prefix = ""
                 print_color_suffix = ""
        except:
            print_color_prefix = ""
            print_color_suffix = ""
            
        targets_display = normalized_target_ans_single
        if question_type == "vqa_standard" and isinstance(direct_accuracy_results["normalized_target_answers"], list):
            targets_display = direct_accuracy_results["normalized_target_answers"]
            if direct_accuracy_results["vqa_score"] is not None:
                 targets_display = f"{targets_display} (VQA Score: {direct_accuracy_results['vqa_score']:.2f})"

        print(f"{print_color_prefix}      (check.py) QType: {question_type}, Model: '{cleaned_model_ans}', Target(s): '{targets_display}', Correct: {is_correct}, Note: {match_method_note}{print_color_suffix}")

    final_note_summary = match_method_note.split('-')[0].strip() if '-' in match_method_note else match_method_note
    if direct_accuracy_results["notes"] and "Skipped" in direct_accuracy_results["notes"]:
        final_note_summary = direct_accuracy_results["notes"]
    
    current_is_correct_status = direct_accuracy_results.get('is_correct', False)
    vqa_score_info = ""
    if direct_accuracy_results.get("vqa_score") is not None:
        vqa_score_info = f", VQA Score: {direct_accuracy_results['vqa_score']:.2f}"

    grades_output = [f"[Direct Comparison ({question_type}): {'Correct' if current_is_correct_status else 'Incorrect'}{vqa_score_info} - {final_note_summary}]"]
    majority_vote_output = f"Direct comparison ({question_type}): {'Correct' if current_is_correct_status else 'Incorrect'}{vqa_score_info}"

    return direct_accuracy_results, grades_output, majority_vote_output