# THIẾT KẾ CHI TIẾT: SYNTHESIZER LOGIC ENGINE

## I. Tổng Quan và Nguyên Tắc Thiết Kế

`SynthesizerEngine` là một engine thuật toán, **không phải LLM**, được thiết kế để đưa ra quyết định cuối cùng một cách có thể dự đoán và kiểm chứng được. Nhiệm vụ duy nhất của nó là tổng hợp các bằng chứng và giả thuyết để đưa ra một câu trả lời duy nhất.

### Nguyên Tắc Cốt Lõi:
1.  **Tính Xác Định (Deterministic):** Với cùng một đầu vào, engine luôn tạo ra cùng một đầu ra.
2.  **Tính Có Thể Kiểm Chứng (Verifiable):** Toàn bộ logic đủ đơn giản để con người có thể kiểm tra và xác nhận.
3.  **Tính Module Hóa (Modular):** Chỉ thực hiện một nhiệm vụ: tổng hợp logic.
4.  **Tính Bất Bại (Always Answers):** Engine được thiết kế để **luôn luôn** đưa ra một câu trả lời cuối cùng, không bao giờ trả về `null`.

---

## II. Định Dạng Dữ Liệu

### Đầu Vào (Inputs) của hàm `synthesize`

| Tên Tham Số | Kiểu Dữ Liệu | Vai Trò | Ví dụ |
| :--- | :--- | :--- | :--- |
| `evidence_set` | `List[Dict]` | Danh sách các bằng chứng đã được `Verifier` xác minh. | `[{"evidence_id": "E01", "answer": "Yes"}]` |
| `hypothesis_set` | `List[Dict]` | Danh sách các quy tắc logic do `Strategist` tạo ra. | `[{"IF": ..., "THEN": ..., "confidence_source": 0.8}]` |
| `answer_candidates`| `List[str]` | Danh sách các câu trả lời ứng viên ban đầu từ `Verifier`. | `["Yes", "no"]` |

### Đầu Ra (Output)

Hàm `synthesize` trả về một đối tượng `Dict` duy nhất với cấu trúc:

```json
{
  "status": "CONCLUSIVE", // Xem các trạng thái bên dưới
  "answer": "Yes",
  "causal_trace": [
    {
      "hypothesis_id": "H_Is_Snowing",
      "triggered_by_evidence": ["E01", "E02"]
    }
  ]
}
```

#### Các Trạng Thái (Status) Có Thể Có:

| Trạng Thái | Ý Nghĩa |
| :--- | :--- |
| `CONCLUSIVE` | Kết luận được đưa ra một cách logic và không có mâu thuẫn. |
| `CONCLUSIVE_AFTER_CONFLICT` | Có mâu thuẫn logic, nhưng đã được giải quyết bằng cách chọn giả thuyết có độ tin cậy (`confidence`) cao nhất. |
| `CONCLUSIVE_BY_FALLBACK` | Không có giả thuyết logic nào được kích hoạt, hệ thống đã phải dùng đến phương án dự phòng. |
| `ERROR` | Có lỗi xảy ra trong quá trình xử lý. |

---

## III. Kiến Trúc Ra Quyết Định 3 Tầng

Đây là kiến trúc cốt lõi của `SynthesizerEngine`, được implement trong hàm `_evaluate_and_decide`. Nó đảm bảo rằng luôn có một câu trả lời được đưa ra.

![Synthesizer Logic Flow](https://i.imgur.com/your_diagram_here.png)
*Sơ đồ luồng logic của engine*

### Tầng 1: Kết Luận Logic Tuyệt Đối (`CONCLUSIVE`)
-   **Kích hoạt khi:** Chỉ có **một kết luận duy nhất** được suy ra từ tất cả các giả thuyết được kích hoạt.
-   **Ví dụ:** Nếu chỉ có giả thuyết `H_Is_Yes` được kích hoạt, và nó kết luận là "Yes".
-   **Hành động:** Trả về kết luận "Yes" với trạng thái `CONCLUSIVE`. Đây là trường hợp lý tưởng nhất.

```python
# Tầng 1: Logic
if len(triggered_hypotheses) > 0:
    unique_conclusions = {h['THEN']['final_answer'] for h in triggered_hypotheses}
    if len(unique_conclusions) == 1:
        final_answer = unique_conclusions.pop()
        # Trả về kết quả...
```

### Tầng 2: Giải Quyết Xung Đột Dựa Trên Confidence (`CONCLUSIVE_AFTER_CONFLICT`)
-   **Kích hoạt khi:** Có **nhiều hơn một kết luận khác nhau** được suy ra (ví dụ: một giả thuyết kết luận "Yes", một giả thuyết khác kết luận "no").
-   **Hành động:**
    1.  Engine sẽ so sánh giá trị `confidence_source` trong mỗi giả thuyết được kích hoạt (giá trị này được `Strategist` cung cấp).
    2.  Nó sẽ chọn kết luận (`final_answer`) từ giả thuyết có `confidence_source` **cao nhất**.
    3.  Nếu có nhiều giả thuyết cùng có `confidence_source` cao nhất nhưng lại đưa ra các kết luận khác nhau (một trường hợp "hòa" thực sự), engine sẽ từ bỏ và chuyển sang Tầng 3.
-   **Mục đích:** Sử dụng `confidence` như một "lá phiếu vàng" để phá vỡ thế bế tắc logic.

```python
# Tầng 2: Logic
best_hypothesis = max(triggered_hypotheses, key=lambda h: h.get('confidence_source', 0))
final_answer = best_hypothesis['THEN']['final_answer']
# ... kiểm tra trường hợp hòa ...
# Trả về kết quả...
```

### Tầng 3: Lựa Chọn "An Toàn" Nhất (`CONCLUSIVE_BY_FALLBACK`)
-   **Kích hoạt khi:**
    -   Không có giả thuyết logic nào được kích hoạt ở Tầng 1.
    -   Xảy ra một cuộc xung đột không thể giải quyết ở Tầng 2.
-   **Hành động:**
    1.  Engine sẽ lấy danh sách `answer_candidates` ban đầu do `Verifier` cung cấp.
    2.  Nó sẽ chọn **ứng viên đầu tiên** (`answer_candidates[0]`) làm câu trả lời cuối cùng.
    3.  Nếu ngay cả `answer_candidates` cũng rỗng, nó sẽ trả về "Unavailable".
-   **Mục đích:** Đảm bảo hệ thống không bao giờ "bó tay", luôn đưa ra một câu trả lời dựa trên phỏng đoán ban đầu tốt nhất.

```python
# Tầng 3: Logic
if answer_candidates:
    final_answer = answer_candidates[0]
    # Trả về kết quả...
else:
    final_answer = "Unavailable"
    # Trả về kết quả...
```

---

## IV. Các Hàm Hỗ Trợ

-   **`synthesize(evidence_set, hypothesis_set, answer_candidates)`**: Hàm chính, điều phối toàn bộ quá trình.
-   **`_build_evidence_map(evidence_set)`**: Tạo một dictionary để tra cứu nhanh các bằng chứng, tối ưu hóa hiệu năng.
-   **`_validate_hypothesis_format(hypothesis)`**: Kiểm tra xem một giả thuyết có tuân thủ đúng định dạng dữ liệu hay không.
-   **`_check_hypothesis_conditions(hypothesis, evidence_map)`**: Duyệt qua các điều kiện `IF` của một giả thuyết và kiểm tra xem chúng có được đáp ứng bởi `evidence_set` hay không.

---

## V. Khả Năng Cải Thiện Trong Tương Lai

Mặc dù engine hiện tại đã rất mạnh mẽ, có một số hướng có thể cải thiện:

1.  **Logic Giải Quyết Xung Đột Phức Tạp Hơn:** Thay vì chỉ dựa vào `confidence` cao nhất, có thể tính tổng `confidence` cho mỗi kết luận. Ví dụ: nếu có 2 giả thuyết cùng kết luận "Yes" với confidence là 0.7 và 0.6, tổng điểm sẽ là 1.3, có thể sẽ thắng một giả thuyết kết luận "no" với confidence 0.8.
2.  **Trọng Số Hóa Bằng Chứng:** Gán trọng số cho các loại bằng chứng khác nhau. Ví dụ, một bằng chứng từ việc nhận dạng vật thể có thể đáng tin cậy hơn một bằng chứng từ việc suy luận về mối quan hệ.
3.  **Học Cách Fallback:** Thay vì luôn chọn ứng viên đầu tiên, Tầng 3 có thể được cải thiện để chọn ứng viên có tần suất xuất hiện cao nhất trong các bộ dữ liệu tương tự.
4.  **Phản Hồi Về Cho `Strategist`:** Nếu engine thường xuyên phải dùng đến Tầng 2 hoặc 3, đây là một tín hiệu cho thấy `Strategist` đang tạo ra các giả thuyết kém chất lượng. Hệ thống có thể ghi nhận thông tin này để tự động cải thiện prompt của `Strategist` trong tương lai. 