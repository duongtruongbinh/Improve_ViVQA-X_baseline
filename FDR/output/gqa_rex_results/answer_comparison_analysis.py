#!/usr/bin/env python3
"""
Phân tích cách thức so sánh Answer với Ground Truth trong GQA-REX Pipeline

Script này sẽ:
1. Giải thích logic so sánh
2. Hiển thị các trường hợp cụ thể 
3. Phân tích các edge cases và vấn đề tiềm ẩn
"""

import json
import re
from collections import Counter, defaultdict

def analyze_answer_comparison():
    """Phân tích chi tiết cách thức so sánh answer với ground_truth"""
    
    print("="*80)
    print("PHÂN TÍCH CÁCH THỨC SO SÁNH ANSWER VỚI GROUND_TRUTH")
    print("="*80)
    
    # Load dữ liệu
    print("📊 Đang đọc file summary_results.json...")
    with open('/home/huytd/multi-agent/multi-agent/FDR/output/gqa_rex_results/summary_results.json', 'r') as f:
        data = json.load(f)
    
    print("\n🔍 LOGIC SO SÁNH HIỆN TẠI:")
    print("-" * 50)
    print("1. Normalize cả predicted_answer và ground_truth bằng _normalize_answer()")
    print("2. So sánh bằng: predicted_answer.lower() == normalized_ground_truth.lower()")
    print()
    print("📝 Hàm _normalize_answer() thực hiện:")
    print("   • Loại bỏ whitespace: answer.strip()")
    print("   • Loại bỏ dấu câu cuối: rstrip('.!?')")
    print("   • Xử lý đặc biệt cho 'yes'/'no': chuyển về lowercase")
    print("   • Giữ nguyên case cho các answer khác")
    
    # Phân tích các trường hợp cụ thể
    analyze_specific_cases(data['results'])
    
    # Tìm các edge cases
    find_edge_cases(data['results'])
    
    # Phân tích lỗi normalization
    analyze_normalization_issues(data['results'])

def _normalize_answer_demo(answer: str) -> str:
    """Demo hàm normalize như trong code gốc"""
    if not answer:
        return answer
    
    # Remove trailing punctuation and extra whitespace  
    normalized = answer.strip().rstrip('.!?').strip()
    
    # Handle common format variations
    if normalized.lower() in ['yes', 'no']:
        return normalized.lower()
    
    # For other answers, keep original case but remove trailing punctuation
    return normalized

def analyze_specific_cases(results):
    """Phân tích các trường hợp cụ thể của so sánh"""
    
    print("\n🎯 PHÂN TÍCH CÁC TRƯỜNG HỢP CỤ THỂ:")
    print("-" * 50)
    
    # Thu thập các mẫu
    comparison_cases = []
    case_types = defaultdict(list)
    
    for result in results[:50]:  # Lấy 50 mẫu đầu để phân tích
        predicted = result.get('predicted_answer', '')
        ground_truth = result.get('ground_truth', '')
        final_answer = result.get('final_answer', '')
        is_correct = result.get('is_correct', False)
        
        # Áp dụng normalize
        normalized_predicted = _normalize_answer_demo(final_answer) if final_answer else predicted
        normalized_gt = _normalize_answer_demo(ground_truth)
        
        # Tính toán is_correct theo logic gốc
        computed_correct = normalized_predicted.lower() == normalized_gt.lower()
        
        case_info = {
            'question_id': result.get('question_id'),
            'question': result.get('question', '')[:60] + "...",
            'final_answer': final_answer,
            'predicted_answer': predicted,
            'ground_truth': ground_truth,
            'normalized_predicted': normalized_predicted,
            'normalized_gt': normalized_gt,
            'is_correct': is_correct,
            'computed_correct': computed_correct,
            'match': is_correct == computed_correct
        }
        
        comparison_cases.append(case_info)
        
        # Phân loại cases
        if predicted.lower() in ['yes', 'no'] and ground_truth.lower() in ['yes', 'no']:
            case_types['yes_no'].append(case_info)
        elif predicted.isdigit() or ground_truth.isdigit():
            case_types['numeric'].append(case_info)
        elif len(predicted.split()) > 1 or len(ground_truth.split()) > 1:
            case_types['multi_word'].append(case_info)
        else:
            case_types['single_word'].append(case_info)
    
    # Hiển thị ví dụ cho từng loại
    print("\n📋 CÁC LOẠI TRƯỜNG HỢP:")
    
    print(f"\n1️⃣  YES/NO Questions ({len(case_types['yes_no'])} cases)")
    for case in case_types['yes_no'][:3]:
        status = "✅" if case['is_correct'] else "❌"
        print(f"   {status} '{case['final_answer']}' vs '{case['ground_truth']}' → {case['is_correct']}")
    
    print(f"\n2️⃣  Numeric Answers ({len(case_types['numeric'])} cases)")  
    for case in case_types['numeric'][:3]:
        status = "✅" if case['is_correct'] else "❌"
        print(f"   {status} '{case['final_answer']}' vs '{case['ground_truth']}' → {case['is_correct']}")
    
    print(f"\n3️⃣  Multi-word Answers ({len(case_types['multi_word'])} cases)")
    for case in case_types['multi_word'][:3]:
        status = "✅" if case['is_correct'] else "❌"
        print(f"   {status} '{case['final_answer']}' vs '{case['ground_truth']}' → {case['is_correct']}")
    
    print(f"\n4️⃣  Single Word Answers ({len(case_types['single_word'])} cases)")
    for case in case_types['single_word'][:3]:
        status = "✅" if case['is_correct'] else "❌"
        print(f"   {status} '{case['final_answer']}' vs '{case['ground_truth']}' → {case['is_correct']}")

def find_edge_cases(results):
    """Tìm các edge cases trong so sánh"""
    
    print("\n🔍 TÌM KIẾM EDGE CASES:")
    print("-" * 50)
    
    edge_cases = {
        'case_mismatch': [],
        'punctuation_issues': [],
        'plural_singular': [],
        'synonym_issues': [],
        'whitespace_issues': []
    }
    
    for result in results:
        predicted = result.get('final_answer', '')
        ground_truth = result.get('ground_truth', '')
        is_correct = result.get('is_correct', False)
        
        if not predicted or not ground_truth:
            continue
        
        # Case mismatch (nhưng nội dung giống)
        if predicted.lower() == ground_truth.lower() and predicted != ground_truth and not is_correct:
            edge_cases['case_mismatch'].append({
                'predicted': predicted,
                'ground_truth': ground_truth,
                'question': result.get('question', '')[:50] + "..."
            })
        
        # Punctuation issues
        if (predicted.rstrip('.!?').lower() == ground_truth.rstrip('.!?').lower() 
            and predicted != ground_truth and not is_correct):
            edge_cases['punctuation_issues'].append({
                'predicted': predicted,
                'ground_truth': ground_truth,
                'question': result.get('question', '')[:50] + "..."
            })
        
        # Plural vs Singular
        predicted_clean = predicted.lower().rstrip('s')
        gt_clean = ground_truth.lower().rstrip('s')
        if (predicted_clean == gt_clean and predicted.lower() != ground_truth.lower() 
            and not is_correct):
            edge_cases['plural_singular'].append({
                'predicted': predicted,
                'ground_truth': ground_truth,
                'question': result.get('question', '')[:50] + "..."
            })
        
        # Whitespace issues
        if predicted.replace(' ', '') == ground_truth.replace(' ', '') and not is_correct:
            edge_cases['whitespace_issues'].append({
                'predicted': f"'{predicted}'",
                'ground_truth': f"'{ground_truth}'",
                'question': result.get('question', '')[:50] + "..."
            })
    
    # Hiển thị edge cases
    for case_type, cases in edge_cases.items():
        if cases:
            print(f"\n🚨 {case_type.upper()} ({len(cases)} cases):")
            for case in cases[:3]:  # Hiển thị tối đa 3 ví dụ
                print(f"   • Predicted: {case['predicted']}")
                print(f"     Ground Truth: {case['ground_truth']}")
                print(f"     Question: {case['question']}")
                print()

def analyze_normalization_issues(results):
    """Phân tích các vấn đề với normalization"""
    
    print("\n⚠️  PHÂN TÍCH VẤN ĐỀ NORMALIZATION:")
    print("-" * 50)
    
    issues = []
    inconsistent_cases = []
    
    for result in results:
        predicted = result.get('final_answer', '')
        ground_truth = result.get('ground_truth', '')
        is_correct = result.get('is_correct', False)
        
        if not predicted or not ground_truth:
            continue
        
        # Simulate normalization
        norm_predicted = _normalize_answer_demo(predicted)
        norm_gt = _normalize_answer_demo(ground_truth)
        
        # Check if normalization logic matches actual result
        expected_correct = norm_predicted.lower() == norm_gt.lower()
        
        if expected_correct != is_correct:
            inconsistent_cases.append({
                'question_id': result.get('question_id'),
                'predicted': predicted,
                'ground_truth': ground_truth,
                'norm_predicted': norm_predicted,
                'norm_gt': norm_gt,
                'expected_correct': expected_correct,
                'actual_correct': is_correct,
                'question': result.get('question', '')[:60] + "..."
            })
    
    if inconsistent_cases:
        print(f"🔴 Tìm thấy {len(inconsistent_cases)} trường hợp không nhất quán:")
        for case in inconsistent_cases[:5]:
            print(f"\n   Question ID: {case['question_id']}")
            print(f"   Question: {case['question']}")
            print(f"   Predicted: '{case['predicted']}' → '{case['norm_predicted']}'")
            print(f"   Ground Truth: '{case['ground_truth']}' → '{case['norm_gt']}'")
            print(f"   Expected: {case['expected_correct']}, Actual: {case['actual_correct']}")
    else:
        print("✅ Không tìm thấy vấn đề nhất quán trong normalization")

def demonstrate_normalization():
    """Demo thực tế của hàm normalization"""
    
    print("\n💡 DEMO HÀM NORMALIZATION:")
    print("-" * 50)
    
    test_cases = [
        "Yes",
        "yes.",
        "No!",
        "Chair",  
        "chair.",
        "  Red  ",
        "Blue?",
        "two",
        "Two.",
        "skier"
    ]
    
    print("Input → Normalized Output:")
    for test in test_cases:
        normalized = _normalize_answer_demo(test)
        print(f"'{test}' → '{normalized}'")

def main():
    """Hàm chính"""
    
    try:
        analyze_answer_comparison()
        demonstrate_normalization()
        
        print("\n" + "="*80)
        print("KẾT LUẬN:")
        print("="*80)
        print("✅ Logic so sánh đơn giản nhưng hiệu quả:")
        print("   • Normalize cả 2 phía để loại bỏ format differences")
        print("   • So sánh case-insensitive bằng .lower()")
        print("   • Xử lý đặc biệt cho Yes/No questions")
        print()
        print("⚠️  Hạn chế:")
        print("   • Không xử lý synonyms (e.g., 'car' vs 'vehicle')")
        print("   • Không xử lý plural/singular (e.g., 'chair' vs 'chairs')")  
        print("   • Không xử lý số viết bằng chữ (e.g., 'two' vs '2')")
        print()
        print("📊 Tổng thể: Logic so sánh phù hợp với dataset GQA-REX")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
