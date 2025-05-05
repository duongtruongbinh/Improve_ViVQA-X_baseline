import json
import re

# Các thư viện tính metric NLG
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from bert_score import score as bert_score


def clean_text(text):
    """
    Nếu cần, bạn có thể loại bỏ các đoạn ký tự (ví dụ: nội dung trong ngoặc)
    """
    return re.sub(r" \([^)]*\)", "", text)


def get_nlg_scores(references, hypotheses, device='cuda'):
    """
    Tính các metric NLG.

    parameters:
      - references: list các danh sách ground truth explanations.
      - hypotheses: list các lời giải thích dự đoán (chuỗi).
      - device: thiết bị chạy BERTScore.

    Trả về: dict chứa các scores.
    """
    # Chuyển đổi sang định dạng cho pycocoevalcap
    gts = {i: refs for i, refs in enumerate(references)}
    res = {i: [hyp] for i, hyp in enumerate(hypotheses)}

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                scores[m] = s
        else:
            scores[method] = score

    # Tính BERTScore (lấy F1 trung bình)
    P, R, F1 = bert_score(
        hypotheses, [refs[0] for refs in references], lang='vi', device=device)
    scores['BERTScore_F1'] = F1.mean().item()

    return scores


def evaluate(json_file, output_file="evaluation_scores.json", device='cuda'):
    # Đọc file JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_examples = len(data)
    correct_count = 0
    task_score = 0.0

    # Danh sách cho toàn bộ dữ liệu (unfiltered) và cho các ví dụ đúng (filtered)
    all_gt_expls = []     # Unfiltered: list các danh sách ground truth explanation
    all_pred_expls = []   # Unfiltered: list các dự đoán (chuỗi)

    filtered_gt_expls = []    # Chỉ các ví dụ đúng
    filtered_pred_expls = []

    for item in data:
        gt_ans = item['gt_ans'].strip().lower()
        pred_ans = item['pred_ans'].strip().lower()

        # Lấy danh sách explanation ground truth (bạn có thể xử lý thêm nếu cần)
        gt_expls = item['gt_explain']

        # Loại bỏ tiền tố "because" (hoặc "vì") ở lời giải thích dự đoán nếu có
        pred_expl = item['pred_explain']
        if pred_expl.lower().startswith("because"):
            pred_expl = pred_expl[7:].strip()  # loại bỏ "because"
        elif pred_expl.lower().startswith("vì"):
            pred_expl = pred_expl[2:].strip()

        # Nếu cần, bạn có thể áp dụng hàm clean_text để loại bỏ thông tin không cần thiết
        # gt_expls = [clean_text(expl) for expl in gt_expls]
        # pred_expl = clean_text(pred_expl)

        # Thêm vào danh sách unfiltered
        all_gt_expls.append(gt_expls)
        all_pred_expls.append(pred_expl)

        # Kiểm tra nếu dự đoán answer khớp với gt answer thì thêm vào danh sách filtered
        if pred_ans == gt_ans:
            correct_count += 1
            filtered_gt_expls.append(gt_expls)
            filtered_pred_expls.append(pred_expl)
            task_score += 1.0  # hoặc bạn có thể sử dụng điểm số theo trọng số nếu cần

    # Tính accuracy và task score
    accuracy = correct_count / total_examples
    task_score_final = task_score / total_examples

    # Tính các metric cho tập tất cả (unfiltered)
    unfiltered_scores = get_nlg_scores(
        all_gt_expls, all_pred_expls, device=device)

    # Tính các metric cho tập filtered: nếu có ít nhất 1 ví dụ đúng
    if len(filtered_pred_expls) > 0:
        filtered_scores = get_nlg_scores(
            filtered_gt_expls, filtered_pred_expls, device=device)
    else:
        # Nếu không có ví dụ nào đúng, gán các giá trị 0
        filtered_scores = {key: 0.0 for key in unfiltered_scores.keys()}

    # Tính điểm scaled (scaled = unfiltered_score * accuracy)
    scaled_scores = {key: value * accuracy for key,
                     value in unfiltered_scores.items()}

    # In kết quả
    print("Accuracy: {:.4f}".format(accuracy))
    print("Task Score: {:.4f}".format(task_score_final))
    print("\nScores cho toàn bộ giải thích (Unfiltered):")
    for key, value in unfiltered_scores.items():
        print("  {}: {:.4f}".format(key, value))
    print("\nScores cho giải thích đúng (Filtered):")
    for key, value in filtered_scores.items():
        print("  {}: {:.4f}".format(key, value))
    print("\nScaled Scores:")
    for key, value in scaled_scores.items():
        print("  {}: {:.4f}".format(key, value))

    # Lưu kết quả vào file JSON
    results = {
        'accuracy': accuracy,
        'task_score': task_score_final,
        'unfiltered_scores': unfiltered_scores,
        'filtered_scores': filtered_scores,
        'scaled_scores': scaled_scores
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nĐã lưu kết quả vào file: {output_file}")


if __name__ == "__main__":
    evaluate("../../results/ViVQA-X_test_Qwen2.5-VL-7B-Instruct.json",
             output_file="../../results/ViVQA-X_test_Qwen2.5-VL-7B-Instruct_score.json", device='cuda')
