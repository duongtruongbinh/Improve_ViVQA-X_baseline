import argparse
import json
import re
from collections import defaultdict

CONTRACTIONS = {
    'aint': "ain't",
    'arent': "aren't",
    'cant': "can't",
    'couldve': "could've",
    'couldnt': "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    'didnt': "didn't",
    'doesnt': "doesn't",
    'dont': "don't",
    'hadnt': "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    'hasnt': "hasn't",
    'havent': "haven't",
    'hed': "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    'hes': "he's",
    'howd': "how'd",
    'howll': "how'll",
    'hows': "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    'Im': "I'm",
    'Ive': "I've",
    'isnt': "isn't",
    'itd': "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    'itll': "it'll",
    "let's": "let's",
    'maam': "ma'am",
    'mightnt': "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    'mightve': "might've",
    'mustnt': "mustn't",
    'mustve': "must've",
    'neednt': "needn't",
    'notve': "not've",
    'oclock': "o'clock",
    'oughtnt': "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    'shant': "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    'shouldve': "should've",
    'shouldnt': "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd',
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    'somebodyll': "somebody'll",
    'somebodys': "somebody's",
    'someoned': "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    'someonell': "someone'll",
    'someones': "someone's",
    'somethingd': "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    'somethingll': "something'll",
    'thats': "that's",
    'thered': "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    'therere': "there're",
    'theres': "there's",
    'theyd': "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    'theyll': "they'll",
    'theyre': "they're",
    'theyve': "they've",
    'twas': "'twas",
    'wasnt': "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    'weve': "we've",
    'werent': "weren't",
    'whatll': "what'll",
    'whatre': "what're",
    'whats': "what's",
    'whatve': "what've",
    'whens': "when's",
    'whered': "where'd",
    'wheres': "where's",
    'whereve': "where've",
    'whod': "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    'wholl': "who'll",
    'whos': "who's",
    'whove': "who've",
    'whyll': "why'll",
    'whyre': "why're",
    'whys': "why's",
    'wont': "won't",
    'wouldve': "would've",
    'wouldnt': "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    'yall': "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    'youd': "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    'youll': "you'll",
    'youre': "you're",
    'youve': "you've",
}
MANUAL_MAP = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10',
}
ARTICLES = {'a', 'an', 'the'}
PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')',
         '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
COMMA_RE = re.compile(r'(\d)(,)(\d)')
PERIOD_RE = re.compile(r'(?!<=\d)(\.)(?!\d)')


def normalize(text: str) -> str:
    """Lower, strip punctuation/digits/articles, expand contractions."""
    text = text.strip().lower()
    # strip bad commas
    text = COMMA_RE.sub(r'\1\3', text)
    # remove punctuation
    for p in PUNCT:
        text = text.replace(p, ' ')
    # remove stray periods
    text = PERIOD_RE.sub('', text)
    # tokenize and map
    words = []
    for w in text.split():
        w = MANUAL_MAP.get(w, w)
        if w in ARTICLES:
            continue
        words.append(CONTRACTIONS.get(w, w))
    return ' '.join(words)


def vqa_score(pred: str, gts: list) -> float:
    """Compute min(#matches/3, 1) with normalized strings."""
    pred_n = normalize(pred)
    gt_n = [normalize(d['answer']) for d in gts]
    matches = sum(1 for a in gt_n if a == pred_n)
    return min(matches / 3.0, 1.0)


def main(preds_path):
    data = json.load(open(preds_path, 'r', encoding='utf-8'))
    scores = []
    by_type = defaultdict(list)

    for entry in data:
        pred = entry['predict_ans']
        gts = entry['gt_ans']
        ans_type = entry.get('answer_type', 'unknown')
        sc = vqa_score(pred, gts)
        scores.append(sc)
        by_type[ans_type].append(sc)

    overall = sum(scores) / len(scores) * 100
    print(f"Overall VQA Accuracy: {overall:.2f}%")
    for t, lst in by_type.items():
        acc_t = sum(lst) / len(lst) * 100
        print(f"  {t:12s}: {acc_t:.2f}%  ({len(lst)} questions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VQAv2 predictions from model output JSON")
    parser.add_argument(
        "--preds", type=str, required=True,
        help="path to JSON file with model predictions and gt_ans")
    args = parser.parse_args()
    main(args.preds)
