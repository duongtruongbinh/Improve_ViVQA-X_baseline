# THIẾT KẾ GỐC (V0): INTEGRATOR AGENT (DEPRECATED)

## I. Giới Thiệu và Bối Cảnh

Tài liệu này mô tả kiến trúc ban đầu của agent tổng hợp, được gọi là `Integrator`. Thiết kế này đã được thay thế bằng các phiên bản deterministic (V1, V2) để giải quyết các nhược điểm nghiêm trọng về tính ổn định và khả năng kiểm chứng.

Mục đích của tài liệu này là để lưu trữ lịch sử, làm rõ các quyết định thiết kế và cung cấp bối cảnh cho sự phát triển của `Synthesizer Engine` hiện tại.

---

## II. Thuật Toán Gốc

`Integrator` hoạt động dựa trên nguyên tắc sử dụng một mô hình VLM (được gọi là `Responder`) bên trong một vòng lặp để đánh giá từng giả thuyết.

```
1: procedure INTEGRATOR(M, I, Q, Qac)
2:   for H, Φ, Rac in M do
3:     PQ ← Concat(H, Φ, I, Q)
4:     Q∗ ← fR(PQ, I, 1; Vθ)  // Gọi VLM để đưa ra câu trả lời
5:     if y in Qac then
6:       Pool[y].append(Rac[confidence]) // Bỏ phiếu bằng confidence của issue
7:     end if
8:   end for
9:   y∗ ← voting(Pool) // Chọn câu trả lời có tổng điểm cao nhất
10:  return y∗
11: end procedure
```

---

## III. Phân Tích Nhược Điểm và Lý Do Thay Thế

Thiết kế V0 tỏ ra không phù hợp với yêu cầu của pipeline FDR vì những lý do sau:

### 1. Tính Không Xác Định (Non-Deterministic)
- **Vấn đề:** Điểm yếu chí mạng nằm ở dòng 4, nơi `Integrator` gọi một mô hình VLM (`fR`) để đưa ra quyết định. Do bản chất của LLM/VLM, kết quả không thể được đảm bảo giống nhau 100% giữa các lần chạy.
- **Tác động:** Điều này làm cho toàn bộ pipeline trở nên không đáng tin cậy, khó debug và không thể kiểm chứng một cách khoa học, đi ngược lại nguyên tắc cốt lõi về tính xác định (determinism).

### 2. Sử Dụng Confidence Score Nông Cạn
- **Vấn đề:** Cơ chế bỏ phiếu (dòng 6) chỉ sử dụng `Rac[confidence]` - độ tin cậy của một *vấn đề liên quan* (relevant issue). Nó hoàn toàn bỏ qua độ tin cậy của chính giả thuyết (`H.confidence`) và các bằng chứng khác.
- **Tác động:** Trọng số của các phiếu bầu không phản ánh đúng mức độ tin cậy của toàn bộ chuỗi lập luận, làm giảm **tính trung thực (faithfulness)** của kết quả cuối cùng.

### 3. Hiệu Năng Kém và Chi Phí Cao
- **Vấn đề:** Việc gọi một mô hình VLM lớn trong một vòng lặp cho mỗi giả thuyết là cực kỳ chậm và tốn kém về mặt tài nguyên tính toán.
- **Tác động:** Không khả thi để áp dụng trên các bộ dữ liệu lớn như VQA-X hay ViVQA-X.

### 4. Khả Năng Lý Giải Kém
- **Vấn đề:** `Integrator` chỉ trả về câu trả lời cuối cùng (`y∗`). Nó không cung cấp bất kỳ một "dấu vết nhân quả" (causal trace) nào cho thấy tại sao `y∗` lại thắng.
- **Tác động:** Không có đủ thông tin cho `Explanation Generator` để tạo ra một lời giải thích có **tính trung thực và có thể lý giải**. Nó không thể giải thích *tại sao* câu trả lời này được chọn thay vì các câu trả lời khác.

---

## IV. Kết Luận

Do những nhược điểm trên, thiết kế V0 đã được thay thế bằng các phiên bản Synthesizer (V1, V2) để ưu tiên tính xác định. Tuy nhiên, bài học về cơ chế "voting" từ V0 đã truyền cảm hứng cho việc phát triển **Synthesizer V3 (Weighted Voting)**, một thiết kế kết hợp được sự ổn định của V1/V2 và sự linh hoạt trong lập luận của V0. 