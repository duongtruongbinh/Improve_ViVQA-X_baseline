import openai
import base64
import requests # Để kiểm tra URL hình ảnh công khai
import os

# ----- Cấu hình cần chỉnh sửa -----
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"  

# Tên model chính xác đang chạy trên vLLM server
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct" 

IMAGE_INPUT = "/mnt/VLAI_data/COCO_Images/val2014/COCO_val2014_000000564636.jpg" 
QUESTION = "What color is his outfit?"


API_KEY = "EMPTY" 
# -----------------------------------

def encode_image_to_base64(image_path):
    """Mã hóa file ảnh cục bộ sang định dạng base64 data URI."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        mime_type = "image/jpeg" 
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif image_path.lower().endswith(".gif"):
            mime_type = "image/gif"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"
        return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
        return None
    except Exception as e:
        print(f"Lỗi khi mã hóa ảnh: {e}")
        return None

def get_image_url_for_payload(image_input_str):
    """
    Xử lý đầu vào hình ảnh. Nếu là đường dẫn cục bộ, mã hóa sang base64.
    Nếu là URL, kiểm tra và trả về URL đó.
    Đây là cách tương tự hàm process_image_for_vlm_agent của bạn có thể hoạt động.
    """
    if image_input_str.startswith("http://") or image_input_str.startswith("https://"):
        try:
            
            response = requests.head(image_input_str, timeout=10) 
            response.raise_for_status() 
            print(f"Hình ảnh từ URL hợp lệ: {image_input_str}")
            return image_input_str
        except requests.exceptions.RequestException as e:
            print(f"Lỗi: Không thể truy cập URL hình ảnh '{image_input_str}': {e}")
            return None
    elif os.path.exists(image_input_str):
        print(f"Mã hóa hình ảnh cục bộ: {image_input_str}")
        return encode_image_to_base64(image_input_str)
    else:
        print(f"Lỗi: Đường dẫn hình ảnh '{image_input_str}' không tồn tại và cũng không phải là URL hợp lệ.")
        return None


client = openai.OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=API_KEY,
)

image_url_for_payload = get_image_url_for_payload(IMAGE_INPUT)

if image_url_for_payload:
    print(f"\nĐang gửi yêu cầu đến model: {MODEL_NAME} tại {VLLM_BASE_URL}")
    print(f"Câu hỏi: {QUESTION}")
    print(f"Sử dụng hình ảnh: {IMAGE_INPUT} (đã xử lý)")

    messages_payload = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url_for_payload,
                        # "detail": "high" # Có thể thêm nếu model hỗ trợ và bạn muốn chất lượng cao nhất
                    },
                },
            ],
        }
    ]

    try:
        print("\nBắt đầu gọi API...")
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages_payload,
            max_tokens=300,    # Lấy từ ví dụ log của bạn, có thể điều chỉnh
            temperature=0.1,   # Lấy từ ví dụ log của bạn, có thể điều chỉnh
            # top_p=1.0,       # Lấy từ ví dụ log của bạn
            # stop=[],         # Lấy từ ví dụ log của bạn
        )
        response_content = completion.choices[0].message.content
        print("\n--- Phản hồi từ Model ---")
        print(response_content)
        # print("\n--- Thông tin Hoàn thành (Raw) ---")
        # print(completion)

    except openai.APIConnectionError as e:
        print(f"\nLỗi kết nối API: {e}")
        print(f"Hãy đảm bảo server vLLM đang chạy tại '{VLLM_BASE_URL}' và có thể truy cập được.")
        print("Kiểm tra log của server vLLM để biết thêm chi tiết.")
    except openai.APIStatusError as e:
        print(f"\nLỗi trạng thái API (từ server vLLM): {e.status_code}")
        print(f"Nội dung phản hồi lỗi: {e.response.text}")
    except openai.RateLimitError as e:
        print(f"\nLỗi giới hạn tỷ lệ (Rate Limit): {e}")
    except Exception as e:
        import traceback
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")
        # traceback.print_exc()
else:
    print("\nKhông thể xử lý hình ảnh đầu vào. Script không thể gửi yêu cầu.")

print("\nScript kiểm tra đã hoàn thành.")
