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

# Complete Pipeline Usage

## Chạy Pipeline Hoàn Chỉnh

Chúng tôi cung cấp pipeline tự động để chạy toàn bộ quy trình từ Step 1 output đến đánh giá cuối cùng.

### Pipeline Steps:
1. **Step 4**: Generate Hypotheses and Probabilities
2. **Step 5**: Rights Allocation and Voting  
3. **Step 6**: Generate Explanations
4. **Evaluation**: Evaluate Results

### Quick Start:

```bash
# Tạo dữ liệu test
./create_test_data.sh

# Chạy pipeline
./run_pipeline.sh results/test_input.json results/test_output

# Chạy với dữ liệu thực
./run_pipeline.sh path/to/step1_output.json results/my_output
```

### Usage:

```bash
./run_pipeline.sh [input_file] [output_dir] [cleanup]
```

**Parameters:**
- `input_file`: File JSON từ Step 1 
- `output_dir`: Thư mục đầu ra
- `cleanup`: Xóa file trung gian (`true`/`false`)

### Input Format:

```json
[
    {
        "question": "What color is the sky?",
        "image_id": "123",
        "image_name": "image.jpg", 
        "question_id": "123_001",
        "answer": "blue",
        "explanation": ["explanation 1", "explanation 2"],
        "answer_candidates": ["blue", "red", "green"],
        "captions": ["Image caption"]
    }
]
```

### Output Files:
- `step4_hypotheses_output.json`: Hypotheses và probabilities
- `step5_voting_output.json`: Voting pool results
- `step6_explanation_output.json`: Generated explanations
- `evaluation_scores.json`: Final evaluation scores
- `pipeline_log.txt`: Execution log

### Dependencies:

```bash
pip install -r requirements.txt
```

### Example:

```bash
# Tạo test data và chạy pipeline
./create_test_data.sh
./run_pipeline.sh results/test_input.json results/my_output true

# Xem kết quả
cat results/my_output/evaluation_scores.json
```