====================================================================================================
                                      📊 FDR Evaluation Summary
====================================================================================================

## Technical Specifications

**Dataset Configuration:**
- Dataset: VQA-X (English Visual Question Answering with Explanations)
- Data Path: `/mnt/VLAI_data/VQA-X/vqaX_val.json`
- Image Directory: `/mnt/VLAI_data/COCO_Images/val2014`
- Split: Validation set
- Language: English

**Model Architecture:**
- Architecture: Dual-model system with smart task routing
- VLM Model: Qwen2.5-VL-7B-Instruct
  - Path: `/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-VL-7B-Instruct`
  - Port: 9100 (handles vision-related tasks)
  - Tasks: Image analysis, object detection, visual reasoning
- LLM Model: Qwen2.5-7B-Instruct
  - Path: `/mnt/dataset1/pretrained_fm/Qwen_Qwen2.5-7B-Instruct`
  - Port: 9200 (handles text-only tasks)
  - Tasks: Strategy planning, answer synthesis, explanation generation
- Backend: vLLM local inference setup

**Experimental Settings:**
- Configuration File: `FDR/config.yaml`
- Evaluation Scale: 10-point G-Eval scoring system
- Agent Configuration:
  - Verifier: VLM + GroundingDINO + DAM (temp=0.7, max_tokens=1000)
  - Strategist: MVKB construction (temp=0.3, max_tokens=3500)
  - Synthesizer: Weighted voting algorithm with confidence weighting
- Hardware: CUDA_VISIBLE_DEVICES="0,1,2", GPU memory utilization=0.85
- Processing: Batch size=5, max_workers=5, caching enabled

====================================================================================================
                                    📊 VQA-X Evaluation Results - Run 1
====================================================================================================

  OVERALL ASSESSMENT
  -----------------------------------
  Total Questions:               100
  VQA Accuracy:                  78.00%
  G-Eval Score (Avg):            7.80 / 10.0
----------------------------------------------------------------------------------------------------
Metric       | BLEU-1  | BLEU-2  | BLEU-3  | BLEU-4  | METEOR  | ROUGE-L P | ROUGE-L R | ROUGE-L F1 |  CIDEr  |  SPICE  | BERT-P  | BERT-R  | BERT-F1 | SemScore
----------------------------------------------------------------------------------------------------
Explanation  |   0.203 |   0.086 |   0.043 |   0.025 |   0.297 |     0.139 |     0.491 |     0.208 |   0.000 |   0.167 |   0.656 |   0.743 |   0.694 |   0.548
----------------------------------------------------------------------------------------------------

  G-Eval Breakdown (10-point scale):
    Relevance:             8.40 / 10.0
    Coherence:             7.70 / 10.0
    Faithfulness:          7.30 / 10.0

====================================================================================================

✅ Script finished.

====================================================================================================
                                    📊 VQA-X Evaluation Results - Run 2
====================================================================================================

  OVERALL ASSESSMENT
  -----------------------------------
  Total Questions:               1458
  VQA Accuracy:                  85.25%
  G-Eval Score (Avg):            8.10 / 10.0
----------------------------------------------------------------------------------------------------
Metric       | BLEU-1  | BLEU-2  | BLEU-3  | BLEU-4  | METEOR  | ROUGE-L P | ROUGE-L R | ROUGE-L F1 |  CIDEr  |  SPICE  | BERT-P  | BERT-R  | BERT-F1 | SemScore
----------------------------------------------------------------------------------------------------
Explanation  |   0.200 |   0.087 |   0.042 |   0.023 |   0.295 |     0.130 |     0.493 |     0.197 |   0.000 |   0.156 |   0.655 |   0.746 |   0.695 |   0.547
----------------------------------------------------------------------------------------------------

  G-Eval Breakdown (10-point scale):
    Relevance:             8.90 / 10.0
    Coherence:             7.90 / 10.0
    Faithfulness:          7.50 / 10.0

====================================================================================================

✅ Script finished.

====================================================================================================
                                    📊 ViVQA-X Dataset Evaluation Setup
====================================================================================================

## ViVQA-X Evaluation Status

**Dataset Configuration:**
- Dataset: ViVQA-X (Vietnamese Visual Question Answering with Explanations)
- Expected Path: `/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json`
- Configuration: Already defined in `FDR/config.yaml` under `vivqax` section
- Language: Vietnamese
- Architecture: Same dual-model system (VLM + LLM routing)

**Current Status:**
❌ **Dataset Not Available**: The ViVQA-X dataset file is not currently accessible at the expected path `/mnt/VLAI_data/ViVQA-X/ViVQA-X_val.json`.

**Next Steps for ViVQA-X Evaluation:**
1. Verify dataset availability and correct path
2. Ensure proper Vietnamese language support in the dual-model architecture
3. Run evaluation using identical methodology as VQA-X:
   - Same 10-point G-Eval scoring system
   - Same comprehensive metrics (BLEU, ROUGE, METEOR, CIDEr, SPICE, BERTScore)
   - Same dual-model task routing (VLM for vision tasks, LLM for text tasks)
4. Compare performance between English (VQA-X) and Vietnamese (ViVQA-X) datasets

**Technical Notes:**
- Configuration is ready in `FDR/config.yaml` - simply change `active_dataset: "vqax"` to `active_dataset: "vivqax"`
- Dual-model architecture supports multilingual processing
- All evaluation metrics are language-agnostic except for text-based similarity measures

====================================================================================================