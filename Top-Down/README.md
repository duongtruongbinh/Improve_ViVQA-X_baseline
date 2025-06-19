# Framework VQA với Suy luận Top-Down (SIRI)

Dự án này là một triển khai của framework **SIRI (Seeker, Integrator, Responder)** được mô tả trong paper "Towards Top-Down Reasoning". Nó sử dụng một kiến trúc cộng tác đa agent để giải quyết các bài toán Trả lời câu hỏi bằng hình ảnh (Visual Question Answering - VQA) với mục tiêu tăng cường khả năng suy luận và cung cấp kết quả có thể giải thích được.

## Bắt đầu Nhanh

Để hiểu rõ về dự án và bắt đầu sử dụng, vui lòng tham khảo các tài liệu chi tiết dưới đây.

### 1. Hướng dẫn Toàn diện

Tài liệu này là nơi tốt nhất để bắt đầu. Nó giải thích kiến trúc của hệ thống, cách cài đặt môi trường, cấu hình và chạy pipeline.

➡️ **Đọc [Hướng dẫn sử dụng (GUIDE.md)](./DOCUMENTATION_GUIDE.md)**

### 2. Cấu trúc Dự án

Nếu bạn muốn hiểu rõ về vai trò của từng tệp và thư mục trong dự án, hãy tham khảo tài liệu này.

➡️ **Khám phá [Cấu trúc Thư mục (STRUCTURE.md)](./DOCUMENTATION_STRUCTURE.md)**

---

## Tóm tắt Luồng hoạt động

1.  **Cài đặt:** Tạo môi trường Conda bằng `VQA_env.yaml`.
2.  **Cấu hình:**
    -   Tạo tệp `openai_key.txt` và điền API key.
    -   Kiểm tra và cập nhật các đường dẫn dataset trong `configs/vivqa_config.yaml`.
3.  **Chạy:** Thực thi `python3 main.py` từ thư mục gốc của project.
4.  **Xem kết quả:** Kiểm tra các tệp được tạo ra trong thư mục `output/`.

## **Towards Top Down Reasoning**

Official PyTorch implementation for the paper:

> **Towards top-down reasoning: An explainable multi-agent approach for visual question answering (TMM 2025)**.
>
> Zeqing Wang, Wentao Wan, Qiqing Lao, Runmeng Chen, Minjie Lang, Xiao Wang, Keze Wang, Liang Lin.
>
> <a href='https://arxiv.org/pdf/2311.17331'><img src='https://img.shields.io/badge/arXiv-2311.17331-red'></a> 

## Environment Prepare
Please refer to [LAVIS](https://github.com/salesforce/LAVIS). But do not use the official code, we modify the response function to obtain the confidence of the answer candidate. The source of LAVIS has been contained in this repo.

## LLM Results
### New LLM or new vqa dataset
We provide a unify api toolkit in api_tools/request_api_zoo.py, which support:
- Official OpenAI's server
- [Siliconflow](https://siliconflow.cn/zh-cn/)
- [Zhipu](https://open.bigmodel.cn/)
### Used LLM Results
Due to the update of LLM's API, we provide the middle results of the experiments we have conducted. You can download them from [GoogleDriver](https://drive.google.com/file/d/1sxj80Zs0KaQU1yZdx8oozjc8vMewwH7X/view?usp=sharing)

## Running
After setting the API key, or preparing the LLM results, you can run the framework via:

 ```bash
  bash run_gpt.sh
 ``` 
### Other Setting

- We use a multi-process to speed up the LLM revoke, you can modify the num of process in 'step_eval_multi_process_api_zoo.py'
- Download corresponding datasets and set the path in:  step_eval_mutil_process_api_zoo.py, test_for_integration_rights_alloction.py



## **TODO**
- ~~Release main framework code based on LAVIS~~
- ~~Release Corresponding middle results~~
- A more clear codebase with running README.md

## **Acknowledgement**
We heavily borrow the code from
[LAVIS](https://github.com/salesforce/LAVIS),
 and [LLaVA](https://github.com/haotian-liu/LLaVA). Thanks for sharing their code.

## **Citation**

If you find the code useful for your work, please star this repo and consider citing:

```
@misc{wang2025topdownreasoningexplainablemultiagent,
      title={Towards Top-Down Reasoning: An Explainable Multi-Agent Approach for Visual Question Answering}, 
      author={Zeqing Wang and Wentao Wan and Qiqing Lao and Runmeng Chen and Minjie Lang and Xiao Wang and Keze Wang and Liang Lin},
      year={2025},
      eprint={2311.17331},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.17331}, 
}
```