import json
import re
import sys
import os

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
    if len(references) == 0 or len(hypotheses) == 0:
        return {
            'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0,
            'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0, 'SPICE': 0.0,
            'BERTScore_F1': 0.0
        }
    
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
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    scores[m] = s
            else:
                scores[method] = score
        except Exception as e:
            print(f"Error computing {method}: {e}")
            if isinstance(method, list):
                for m in method:
                    scores[m] = 0.0
            else:
                scores[method] = 0.0

    # Tính BERTScore (lấy F1 trung bình)
    try:
        # Lấy explanation đầu tiên từ mỗi reference list
        ref_texts = []
        for refs in references:
            if isinstance(refs, list) and len(refs) > 0:
                ref_texts.append(refs[0])
            else:
                ref_texts.append(str(refs))
        
        P, R, F1 = bert_score(
            hypotheses, ref_texts, lang='en', device=device)  # Sử dụng 'en' vì có thể dữ liệu là tiếng Anh
        scores['BERTScore_F1'] = F1.mean().item()
    except Exception as e:
        print(f"Error computing BERTScore: {e}")
        scores['BERTScore_F1'] = 0.0

    return scores


def evaluate_step6_output(json_file, output_file="step6_evaluation_scores.json", device='cuda'):
    """
    Đánh giá output của step 6 dựa trên:
    - answer vs final_answer (từ voting_results)
    - explanation vs gen_explanation
    """
    
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

    skipped_items = 0

    for item in data:
        try:
            # Lấy ground truth answer và explanation
            gt_ans = str(item['answer']).strip().lower()
            gt_expls = item['explanation']  # Danh sách các explanations
            
            # Lấy predicted answer từ voting_results
            if 'voting_results' in item and 'final_answer' in item['voting_results']:
                pred_ans = str(item['voting_results']['final_answer']).strip().lower()
            else:
                print(f"Warning: No final_answer found for question_id {item.get('question_id', 'unknown')}")
                skipped_items += 1
                continue
            
            # Lấy generated explanation
            if 'gen_explanation' not in item:
                print(f"Warning: No gen_explanation found for question_id {item.get('question_id', 'unknown')}")
                skipped_items += 1
                continue
                
            pred_expl = item['gen_explanation']

            # Loại bỏ tiền tố "The answer is..." từ generated explanation nếu có
            if pred_expl.lower().startswith("the answer is"):
                # Tìm vị trí "because" để bắt đầu phần explanation
                because_pos = pred_expl.lower().find("because")
                if because_pos != -1:
                    pred_expl = pred_expl[because_pos + 7:].strip()  # loại bỏ "because"
                else:
                    # Nếu không có "because", cố gắng tách phần sau dấu cách đầu tiên sau "The answer is"
                    parts = pred_expl.split(" ", 4)  # Tách thành tối đa 5 phần
                    if len(parts) > 4:
                        pred_expl = parts[4].strip()
            
            # Loại bỏ dấu ngoặc kép nếu có
            pred_expl = pred_expl.strip('"')

            # Thêm vào danh sách unfiltered
            all_gt_expls.append(gt_expls)
            all_pred_expls.append(pred_expl)

            # Kiểm tra nếu dự đoán answer khớp với gt answer thì thêm vào danh sách filtered
            if pred_ans == gt_ans:
                correct_count += 1
                filtered_gt_expls.append(gt_expls)
                filtered_pred_expls.append(pred_expl)
                task_score += 1.0

        except Exception as e:
            print(f"Error processing item {item.get('question_id', 'unknown')}: {e}")
            skipped_items += 1
            continue

    # Cập nhật total_examples sau khi loại bỏ các item bị skip
    total_examples_processed = total_examples - skipped_items
    
    if total_examples_processed == 0:
        print("Error: No valid examples to process!")
        return
    
    # Tính accuracy và task score
    accuracy = correct_count / total_examples_processed
    task_score_final = task_score / total_examples_processed

    print(f"Total examples: {total_examples}")
    print(f"Skipped examples: {skipped_items}")
    print(f"Processed examples: {total_examples_processed}")
    print(f"Correct predictions: {correct_count}")

    # Tính các metric cho tập tất cả (unfiltered)
    print("Computing unfiltered scores...")
    unfiltered_scores = get_nlg_scores(
        all_gt_expls, all_pred_expls, device=device)

    # Tính các metric cho tập filtered: nếu có ít nhất 1 ví dụ đúng
    if len(filtered_pred_expls) > 0:
        print("Computing filtered scores...")
        filtered_scores = get_nlg_scores(
            filtered_gt_expls, filtered_pred_expls, device=device)
    else:
        # Nếu không có ví dụ nào đúng, gán các giá trị 0
        print("No correct predictions found, setting filtered scores to 0")
        filtered_scores = {key: 0.0 for key in unfiltered_scores.keys()}

    # Tính điểm scaled (scaled = unfiltered_score * accuracy)
    scaled_scores = {key: value * accuracy for key,
                     value in unfiltered_scores.items()}

    # In kết quả
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Task Score: {task_score_final:.4f}")
    
    print(f"\nScores cho toàn bộ giải thích (Unfiltered - {len(all_pred_expls)} examples):")
    for key, value in unfiltered_scores.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nScores cho giải thích đúng (Filtered - {len(filtered_pred_expls)} examples):")
    for key, value in filtered_scores.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nScaled Scores (Unfiltered × Accuracy):")
    for key, value in scaled_scores.items():
        print(f"  {key}: {value:.4f}")

    # Lưu kết quả vào file JSON
    results = {
        'total_examples': total_examples,
        'processed_examples': total_examples_processed,
        'skipped_examples': skipped_items,
        'correct_predictions': correct_count,
        'accuracy': accuracy,
        'task_score': task_score_final,
        'unfiltered_scores': unfiltered_scores,
        'filtered_scores': filtered_scores,
        'scaled_scores': scaled_scores
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nĐã lưu kết quả vào file: {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate step 6 output")
    parser.add_argument("--input_file", default="results/step6_explanation_output.json", help="Input file path")
    parser.add_argument("--output_file", default="results/step6_evaluation_scores.json", help="Output file path")
    parser.add_argument("--device", default="cuda", help="Device for computation (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Kiểm tra xem file input có tồn tại không
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found!")
        print("Please make sure the file path is correct.")
        sys.exit(1)
    
    # Chạy evaluation
    evaluate_step6_output(
        args.input_file,
        output_file=args.output_file,
        device=args.device
    )
