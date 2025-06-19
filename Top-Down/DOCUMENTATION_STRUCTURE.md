# Tài liệu Cấu trúc Thư mục

Tài liệu này mô tả cấu trúc thư mục và vai trò của từng tệp tin trong dự án SIRI VQA.

```
Top-Down/
│
├── configs/
│   └── vivqa_config.yaml
│
├── core/
│   ├── __init__.py
│   ├── agents.py
│   └── pipeline.py
│
├── output/
│   ├── siri_pipeline_results.json
│   └── siri_summary.txt
│
├── main.py
├── openai_key.txt
├── VQA_env.yaml
├── README.md
├── DOCUMENTATION_GUIDE.md
└── DOCUMENTATION_STRUCTURE.md
```

## Mô tả Chi tiết

### `/configs/`
- **Mục đích:** Chứa các tệp cấu hình cho pipeline.
- **`vivqa_config.yaml`**: Tệp cấu hình chính. Tại đây bạn có thể định nghĩa split của dataset, số lượng câu hỏi, model AI, và quan trọng nhất là các đường dẫn đến dữ liệu.

### `/core/`
- **Mục đích:** Chứa logic cốt lõi của ứng dụng.
- **`agents.py`**: Định nghĩa các class cho ba agent chính của kiến trúc SIRI: `ResponderAgent`, `SeekerAgent`, và `IntegratorAgent`. Đây là nơi logic suy luận được triển khai.
- **`pipeline.py`**: Chứa hàm `run_siri_pipeline`, có vai trò điều phối toàn bộ luồng hoạt động. Nó khởi tạo các agent, tải dữ liệu, chạy pipeline trên từng mẫu, và lưu kết quả.

### `/output/`
- **Mục đích:** Thư mục mặc định để lưu trữ tất cả các kết quả đầu ra từ pipeline. Thư mục này được tạo tự động khi chạy.
- **`siri_pipeline_results.json`**: Chứa kết quả chi tiết và dấu vết giải thích được (explainability trace) cho mỗi câu hỏi được xử lý.
- **`siri_summary.txt`**: Chứa bản tóm tắt ngắn gọn về kết quả, bao gồm độ chính xác.

---

### Các tệp ở thư mục gốc (`Top-Down/`)

- **`main.py`**: Điểm khởi đầu (entry point) của ứng dụng. Nó chỉ chịu trách nhiệm phân tích đối số dòng lệnh (đường dẫn tệp config) và gọi hàm `run_siri_pipeline`.

- **`openai_key.txt`**: **(Cần được tạo thủ công)** Tệp này dùng để lưu trữ OpenAI API key của bạn một cách an toàn, tách biệt khỏi mã nguồn.

- **`VQA_env.yaml`**: Tệp định nghĩa môi trường Conda. Nó liệt kê tất cả các thư viện Python cần thiết để chạy dự án, đảm bảo tính tái lập (reproducibility).

- **`README.md`**: Tệp giới thiệu cấp cao nhất của dự án. Cung cấp một bản tóm tắt ngắn gọn và các liên kết đến các tài liệu chi tiết hơn.

- **`DOCUMENTATION_GUIDE.md`**: (Tệp này) Hướng dẫn chi tiết về cách cài đặt, cấu hình, và chạy dự án.

- **`DOCUMENTATION_STRUCTURE.md`**: (Tệp bạn đang đọc) Mô tả cấu trúc thư mục của dự án. 