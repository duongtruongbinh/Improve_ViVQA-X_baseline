# Hướng dẫn sử dụng Framework VQA SIRI

Tài liệu này cung cấp hướng dẫn toàn diện để cài đặt, cấu hình và vận hành framework VQA dựa trên kiến trúc SIRI (Seeker, Integrator, Responder).

## 1. Tổng quan Kiến trúc

Framework này là một triển khai của phương pháp **"Towards Top-Down Reasoning"**, sử dụng một hệ thống cộng tác ba agent để trả lời câu hỏi về hình ảnh. Mục tiêu là mô phỏng lại quá trình suy luận của con người bằng cách chia một câu hỏi lớn thành nhiều vấn đề con liên quan, từ đó tăng cường độ chính xác và cung cấp khả năng giải thích cho câu trả lời.

Kiến trúc bao gồm ba thành phần chính được định nghĩa trong `core/agents.py`:

### a. Responder Agent (Agent Phản hồi)
- **Nền tảng:** Dựa trên một Mô hình Ngôn ngữ-Tầm nhìn (VLM) mạnh mẽ (`gpt-4o-mini`).
- **Nhiệm vụ:**
  1.  **Tạo Phản hồi Ban đầu:** Khi nhận một cặp (Câu hỏi, Ảnh), nó tạo ra một chú thích (caption) cho ảnh và một danh sách các ứng viên câu trả lời tiềm năng (top-k answer candidates).
  2.  **Trả lời Câu hỏi trong Bối cảnh:** Trả lời lại câu hỏi gốc khi được cung cấp thêm một "giả thuyết" từ Integrator.

### b. Seeker Agent (Agent Tìm kiếm)
- **Nền tảng:** Dựa trên một Mô hình Ngôn ngữ Lớn (LLM) (`gpt-4o-mini`).
- **Nhiệm vụ (Cốt lõi của framework):**
  1.  Nhận thông tin ban đầu từ Responder.
  2.  **Tạo "Vấn đề liên quan" (Relevant Issues):** Tạo ra các câu hỏi phụ để giúp phân biệt giữa các ứng viên câu trả lời.
  3.  **Xây dựng "Giả thuyết" (Hypotheses):** Yêu cầu Responder trả lời các vấn đề liên quan, sau đó kết hợp các câu trả lời này để tạo ra các giả thuyết logic dạng "NẾU... THÌ...".
  4.  **Gán Điểm tin cậy (Confidence Score):** Sử dụng kiến thức của LLM để gán một điểm tin cậy cho mỗi giả thuyết.
  5.  **Tạo Cơ sở tri thức đa góc nhìn (MVKB):** Tập hợp tất cả thông tin trên (vấn đề, câu trả lời vấn đề, giả thuyết, điểm tin cậy) vào một cấu trúc dữ liệu duy nhất.

### c. Integrator Agent (Agent Tích hợp)
- **Nền tảng:** Một cơ chế logic, không phải một mô hình AI.
- **Nhiệm vụ:**
  1.  Nhận MVKB từ Seeker.
  2.  **Thực hiện Bỏ phiếu theo trọng số (Weighted Voting):** Yêu cầu Responder "bỏ phiếu" cho các ứng viên câu trả lời ban đầu bằng cách hỏi lại câu hỏi gốc trong bối cảnh của từng giả thuyết trong MVKB.
  3.  Tổng hợp các phiếu bầu (có trọng số là điểm tin cậy của giả thuyết tương ứng) và chọn ra câu trả lời có tổng điểm cao nhất.

## 2. Cài đặt Môi trường

Dự án sử dụng Conda để quản lý môi trường.

**Bước 1: Tạo môi trường Conda**
Từ thư mục gốc của dự án (`/VQA`), chạy lệnh:
```sh
# --force sẽ ghi đè lên môi trường cũ nếu đã tồn tại
conda env create -f Top-Down/VQA_env.yaml --force
```

**Bước 2: Kích hoạt môi trường**
```sh
conda activate VQA_env
```

## 3. Cấu hình

Toàn bộ pipeline được điều khiển bởi một tệp cấu hình duy nhất.

### a. Thiết lập API Key
- Tạo một tệp tin mới tại `Top-Down/openai_key.txt`.
- Dán OpenAI API key của bạn vào tệp này (chỉ chuỗi key, không có gì khác).

### b. Tệp cấu hình `vivqa_config.yaml`
- **Địa điểm:** `Top-Down/configs/vivqa_config.yaml`.
- **Các tham số chính:**
  - `inference.dataset_split`: Chọn split của dataset để chạy ('val', 'rest-val', 'test', etc.).
  - `inference.num_questions`: Số lượng câu hỏi cần xử lý. Đặt là `-1` để chạy toàn bộ split.
  - `model.name`: Tên model OpenAI sẽ được sử dụng cho các agent.
  - `dataset_paths`: **QUAN TRỌNG:** Chứa các đường dẫn **tuyệt đối** đến các tệp dataset và thư mục hình ảnh. Bạn phải đảm bảo các đường dẫn này chính xác trên hệ thống của mình.

## 4. Thực thi Pipeline

Sau khi đã cài đặt và cấu hình, bạn có thể chạy pipeline từ thư mục gốc của dự án (`/VQA`).

```sh
# Chạy với tệp cấu hình mặc định
python3 Top-Down/main.py

# Hoặc chỉ định một tệp cấu hình khác
python3 Top-Down/main.py --config /path/to/your/config.yaml
```

## 5. Phân tích Kết quả

Pipeline sẽ tạo ra hai tệp kết quả trong thư mục `Top-Down/output/`:

### a. `siri_pipeline_results.json`
Tệp này chứa kết quả chi tiết cho từng câu hỏi đã xử lý. Mỗi mục là một đối tượng JSON chứa:
- `question_id`, `question`, `ground_truth_answer`, `final_answer`.
- `is_correct`: `true`/`false` nếu có ground truth để so sánh.
- `explainability_trace`: **Phần quan trọng nhất**, chứa toàn bộ "dấu vết suy luận" của hệ thống, bao gồm:
  - `initial_caption`: Chú thích ban đầu của ảnh.
  - `initial_answer_candidates`: Các ứng viên câu trả lời ban đầu.
  - `multi_view_knowledge_base`: Toàn bộ MVKB mà Seeker đã tạo ra, đây là cơ sở cho quyết định cuối cùng.

### b. `siri_summary.txt`
Một tệp văn bản đơn giản cung cấp tóm tắt về quá trình chạy, bao gồm số lượng câu hỏi đã xử lý và độ chính xác cuối cùng (nếu có thể tính toán). 