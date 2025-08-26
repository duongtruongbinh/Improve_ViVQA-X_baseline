#!/usr/bin/env python3
"""
Phân tích Synthesis Status trong GQA-REX Pipeline

Script này giải thích chi tiết về 3 trạng thái synthesis status:
- CONCLUSIVE
- CONCLUSIVE_AFTER_CONFLICT  
- CONCLUSIVE_BY_FALLBACK
"""

import json
import matplotlib.pyplot as plt
from collections import Counter

def load_results():
    """Load kết quả từ file wrong_answers.json và summary_results.json"""
    
    print("📊 Đang đọc kết quả từ file JSON...")
    
    # Load wrong answers
    with open('/home/huytd/multi-agent/multi-agent/FDR/output/gqa_rex_results/wrong_answers.json', 'r') as f:
        wrong_data = json.load(f)
    
    # Load summary results
    with open('/home/huytd/multi-agent/multi-agent/FDR/output/gqa_rex_results/summary_results.json', 'r') as f:
        summary_data = json.load(f)
    
    return wrong_data, summary_data

def analyze_synthesis_status(wrong_data, summary_data):
    """Phân tích chi tiết synthesis status"""
    
    print("\n" + "="*80)
    print("PHÂN TÍCH SYNTHESIS STATUS TRONG GQA-REX PIPELINE")
    print("="*80)
    
    # Thống kê synthesis status trong tất cả kết quả
    all_status = []
    correct_by_status = {}
    total_by_status = {}
    
    for result in summary_data['results']:
        status = result.get('synthesis_status', 'UNKNOWN')
        is_correct = result.get('is_correct', False)
        
        all_status.append(status)
        
        if status not in total_by_status:
            total_by_status[status] = 0
            correct_by_status[status] = 0
        
        total_by_status[status] += 1
        if is_correct:
            correct_by_status[status] += 1
    
    # In thống kê tổng quan
    print("\n🎯 THỐNG KÊ TỔNG QUAN:")
    print("-" * 50)
    for status in sorted(total_by_status.keys()):
        total = total_by_status[status]
        correct = correct_by_status[status]
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"{status:25} | Tổng: {total:4} | Đúng: {correct:4} | Độ chính xác: {accuracy:5.1f}%")
    
    # Giải thích chi tiết từng status
    print("\n📋 GIẢI THÍCH CHI TIẾT CÁC SYNTHESIS STATUS:")
    print("-" * 80)
    
    print("\n1️⃣  CONCLUSIVE (Kết luận Trực tiếp)")
    print("   • Định nghĩa: Pipeline tìm được MỘT kết luận logic duy nhất")
    print("   • Điều kiện: Chỉ có 1 hypothesis được kích hoạt và đưa ra câu trả lời")
    print("   • Độ tin cậy: CAO - Kết quả dựa trên logic thuần túy")
    print("   • Code: Tier 1 trong _evaluate_and_decide()")
    print("   • Ví dụ: Tất cả evidence đều chỉ về cùng một câu trả lời")
    
    print("\n2️⃣  CONCLUSIVE_AFTER_CONFLICT (Kết luận Sau Xung đột)")
    print("   • Định nghĩa: Pipeline có nhiều kết luận khác nhau, giải quyết bằng confidence")
    print("   • Điều kiện: Nhiều hypothesis được kích hoạt với kết luận khác nhau")
    print("   • Độ tin cậy: TRUNG BÌNH - Chọn hypothesis có confidence cao nhất")
    print("   • Code: Tier 2 trong _evaluate_and_decide()")
    print("   • Ví dụ: Evidence mâu thuẫn, phải chọn câu trả lời tin cậy nhất")
    
    print("\n3️⃣  CONCLUSIVE_BY_FALLBACK (Kết luận Dự phòng)")
    print("   • Định nghĩa: Pipeline không tìm được kết luận logic, dùng fallback")
    print("   • Điều kiện: Không có hypothesis nào được kích hoạt")
    print("   • Độ tin cậy: THẤP - Chọn answer_candidate đầu tiên hoặc 'Unavailable'")
    print("   • Code: Tier 3 trong _evaluate_and_decide()")
    print("   • Ví dụ: Evidence không đủ hoặc không khớp với bất kỳ hypothesis nào")
    
    # Phân tích mối quan hệ với độ chính xác
    print("\n📈 PHÂN TÍCH MỐI QUAN HỆ VỚI ĐỘ CHÍNH XÁC:")
    print("-" * 50)
    
    accuracies = [(status, (correct_by_status[status]/total_by_status[status])*100) 
                  for status in total_by_status.keys()]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for i, (status, acc) in enumerate(accuracies, 1):
        print(f"{i}. {status}: {acc:.1f}% độ chính xác")
    
    # Phân tích lỗi theo synthesis status
    print("\n❌ PHÂN TÍCH LỖI THEO SYNTHESIS STATUS:")
    print("-" * 50)
    
    wrong_status_count = Counter()
    for result in wrong_data['wrong_answers']:
        status = result.get('synthesis_status', 'UNKNOWN')
        wrong_status_count[status] += 1
    
    total_wrong = sum(wrong_status_count.values())
    
    for status, count in wrong_status_count.most_common():
        percentage = (count / total_wrong) * 100
        print(f"{status:25} | {count:4} lỗi ({percentage:5.1f}% tổng số lỗi)")
    
    return total_by_status, correct_by_status, wrong_status_count

def create_visualization(total_by_status, correct_by_status, wrong_status_count):
    """Tạo visualization cho synthesis status"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Phân tích Synthesis Status - GQA-REX Pipeline', fontsize=16, fontweight='bold')
    
    # 1. Phân bố synthesis status
    status_names = list(total_by_status.keys())
    status_counts = [total_by_status[s] for s in status_names]
    
    colors = ['#2E8B57', '#FF6347', '#4169E1']  # SeaGreen, Tomato, RoyalBlue
    
    ax1.pie(status_counts, labels=status_names, autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Phân bố Synthesis Status\n(Tất cả 5000 samples)', fontweight='bold')
    
    # 2. Độ chính xác theo status
    accuracies = [(correct_by_status[s]/total_by_status[s])*100 for s in status_names]
    
    bars = ax2.bar(status_names, accuracies, color=colors, alpha=0.7)
    ax2.set_title('Độ chính xác theo Synthesis Status', fontweight='bold')
    ax2.set_ylabel('Độ chính xác (%)')
    ax2.set_ylim(0, 100)
    
    # Thêm giá trị lên thanh
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Xoay nhãn để dễ đọc
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Phân bố lỗi theo status
    wrong_status_names = list(wrong_status_count.keys())
    wrong_counts = [wrong_status_count[s] for s in wrong_status_names]
    
    ax3.bar(wrong_status_names, wrong_counts, color=['#FF6347', '#2E8B57', '#4169E1'], alpha=0.7)
    ax3.set_title('Số lượng lỗi theo Synthesis Status', fontweight='bold')
    ax3.set_ylabel('Số lượng lỗi')
    ax3.tick_params(axis='x', rotation=45)
    
    # Thêm giá trị lên thanh
    for i, (name, count) in enumerate(zip(wrong_status_names, wrong_counts)):
        ax3.text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')
    
    # 4. So sánh đúng vs sai
    width = 0.35
    x_pos = range(len(status_names))
    
    correct_counts = [correct_by_status[s] for s in status_names]
    wrong_counts_aligned = [total_by_status[s] - correct_by_status[s] for s in status_names]
    
    ax4.bar([x - width/2 for x in x_pos], correct_counts, width, label='Đúng', color='#2E8B57', alpha=0.7)
    ax4.bar([x + width/2 for x in x_pos], wrong_counts_aligned, width, label='Sai', color='#FF6347', alpha=0.7)
    
    ax4.set_title('So sánh Kết quả Đúng vs Sai', fontweight='bold')
    ax4.set_ylabel('Số lượng')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(status_names)
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Lưu visualization
    output_path = '/home/huytd/multi-agent/multi-agent/FDR/output/gqa_rex_results/synthesis_status_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Visualization đã được lưu tại: {output_path}")
    
    plt.show()

def show_examples():
    """Hiển thị ví dụ cụ thể cho từng synthesis status"""
    
    print("\n" + "="*80)
    print("VÍ DỤ CỤ THỂ CHO TỪNG SYNTHESIS STATUS")
    print("="*80)
    
    # Load dữ liệu
    with open('/home/huytd/multi-agent/multi-agent/FDR/output/gqa_rex_results/summary_results.json', 'r') as f:
        data = json.load(f)
    
    # Tìm ví dụ cho từng status
    examples = {}
    for result in data['results']:
        status = result.get('synthesis_status', 'UNKNOWN')
        if status not in examples:
            examples[status] = []
        if len(examples[status]) < 2:  # Lấy tối đa 2 ví dụ mỗi loại
            examples[status].append(result)
    
    for status in ['CONCLUSIVE', 'CONCLUSIVE_AFTER_CONFLICT', 'CONCLUSIVE_BY_FALLBACK']:
        if status in examples:
            print(f"\n🔍 VÍ DỤ CHO {status}:")
            print("-" * 60)
            
            for i, example in enumerate(examples[status][:1], 1):  # Chỉ hiển thị 1 ví dụ
                print(f"\nVí dụ {i}:")
                print(f"  Question: {example.get('question', 'N/A')}")
                print(f"  Ground Truth: {example.get('ground_truth', 'N/A')}")
                print(f"  Final Answer: {example.get('final_answer', 'N/A')}")
                print(f"  Correct: {example.get('is_correct', 'N/A')}")
                print(f"  Explanation: {example.get('explanation', 'N/A')[:100]}...")

def main():
    """Hàm chính"""
    
    try:
        # Load dữ liệu
        wrong_data, summary_data = load_results()
        
        # Phân tích synthesis status
        total_by_status, correct_by_status, wrong_status_count = analyze_synthesis_status(wrong_data, summary_data)
        
        # Tạo visualization
        create_visualization(total_by_status, correct_by_status, wrong_status_count)
        
        # Hiển thị ví dụ
        show_examples()
        
        print("\n✅ Phân tích hoàn tất!")
        print("\nTóm tắt:")
        print("- CONCLUSIVE: Kết luận logic trực tiếp (độ tin cậy cao)")
        print("- CONCLUSIVE_AFTER_CONFLICT: Giải quyết xung đột bằng confidence (độ tin cậy trung bình)")
        print("- CONCLUSIVE_BY_FALLBACK: Sử dụng fallback (độ tin cậy thấp)")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
