#!/bin/bash

# Multi-Agent VQA Pipeline Runner
# Runs the complete pipeline from hypothesis generation to evaluation
# Author: Pipeline Automation Script
# Date: $(date)

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# CUDA Device Configuration (can be overridden by environment variable)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE:-1}

# Environment Configuration (can be overridden by environment variable)
CONDA_ENV_NAME=${CONDA_ENV:-"ma_vqa"}

# Number of samples to process (can be overridden by environment variable)
NUM_SAMPLES=${NUM_SAMPLES:-100}

# Dataset configuration (can be overridden by environment variable)
DATASET=${DATASET:-"vqa"}
PROB2WORD_TYPE=${PROB2WORD_TYPE:-"5"}

# Base directory
BASE_DIR="/home/huytd/multi-agent/multi-agent/top_down_baseline"

# Input/Output file paths
INPUT_FILE="${BASE_DIR}/results/step1_vlm_vqax_output.json"
STEP4_OUTPUT="${BASE_DIR}/results/integrated_step4_step4_2_output.json"
STEP5_OUTPUT="${BASE_DIR}/results/step5_voting_pool_integration_output.json"
STEP6_OUTPUT="${BASE_DIR}/results/step6_explanation_output.json"
EVALUATION_OUTPUT="${BASE_DIR}/results/step6_evaluation_scores.json"

# Prompt files
PROBABILITY_PROMPTS="${BASE_DIR}/prompts/probability.prompts"
TRANSFORM_PROMPTS="${BASE_DIR}/prompts/transfor_statement.prompt"
EXPLANATION_PROMPT="${BASE_DIR}/prompts/explanation.prompts"

# Python scripts
STEP4_SCRIPT="${BASE_DIR}/step4_step4_2_integrated.py"
STEP5_SCRIPT="${BASE_DIR}/test_for_integration_rights_alloction.py"
STEP6_SCRIPT="${BASE_DIR}/step6_explanation.py"
EVALUATION_SCRIPT="${BASE_DIR}/evaluate_step6_output.py"

# =============================================================================
# FUNCTIONS
# =============================================================================

# Function to print colored output
print_header() {
    echo ""
    echo "=================================================================="
    echo "$1"
    echo "=================================================================="
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: File not found: $1"
        exit 1
    fi
}

# Function to activate conda environment
activate_env() {
    print_step "Activating conda environment: $CONDA_ENV_NAME"
    
    # Initialize conda for bash shell
    eval "$(conda shell.bash hook)"
    
    # Activate environment
    conda activate $CONDA_ENV_NAME
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully activated environment: $CONDA_ENV_NAME"
        echo "Current Python: $(which python)"
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    else
        echo "ERROR: Failed to activate conda environment: $CONDA_ENV_NAME"
        exit 1
    fi
}

# Function to create results directory
setup_directories() {
    print_step "Setting up directories"
    mkdir -p "${BASE_DIR}/results"
    echo "✓ Results directory ready"
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

print_header "MULTI-AGENT VQA PIPELINE"
start_time=$(date +%s)
echo "Pipeline execution started at: $(date)"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Environment: $CONDA_ENV_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Base Directory: $BASE_DIR"

# Setup
setup_directories
activate_env

# Check input file exists
print_step "Checking input file"
check_file "$INPUT_FILE"
echo "✓ Input file found: $INPUT_FILE"

# Check all required script files
print_step "Checking script files"
check_file "$STEP4_SCRIPT"
check_file "$STEP5_SCRIPT"
check_file "$STEP6_SCRIPT"
check_file "$EVALUATION_SCRIPT"
echo "✓ All script files found"

# Check prompt files
print_step "Checking prompt files"
check_file "$PROBABILITY_PROMPTS"
check_file "$TRANSFORM_PROMPTS"
check_file "$EXPLANATION_PROMPT"
echo "✓ All prompt files found"

# =============================================================================
# STEP 1: Hypothesis Generation and Probability Assessment (Step 4 & 4.2)
# =============================================================================

print_header "STEP 1: HYPOTHESIS GENERATION & PROBABILITY ASSESSMENT"
print_step "Running integrated step4_step4_2..."

python "$STEP4_SCRIPT" \
    --input_file "$INPUT_FILE" \
    --output_file "$STEP4_OUTPUT" \
    --probability_prompts "$PROBABILITY_PROMPTS" \
    --transform_prompts "$TRANSFORM_PROMPTS" \
    --num_samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✓ Step 4 & 4.2 completed successfully"
    check_file "$STEP4_OUTPUT"
    echo "✓ Output file generated: $STEP4_OUTPUT"
else
    echo "ERROR: Step 4 & 4.2 failed"
    exit 1
fi

# =============================================================================
# STEP 2: Voting Pool Integration (Step 5)
# =============================================================================

print_header "STEP 2: VOTING POOL INTEGRATION"
print_step "Running voting pool integration..."

python "$STEP5_SCRIPT" \
    --dataset "$DATASET" \
    --prob2word_type "$PROB2WORD_TYPE" \
    --prompt_maker_name "only_if_prompt_make" \
    --input_file "$STEP4_OUTPUT" \
    --output_file "$STEP5_OUTPUT"

if [ $? -eq 0 ]; then
    echo "✓ Step 5 (Voting Pool) completed successfully"
    check_file "$STEP5_OUTPUT"
    echo "✓ Output file generated: $STEP5_OUTPUT"
else
    echo "ERROR: Step 5 (Voting Pool) failed"
    exit 1
fi

# =============================================================================
# STEP 3: Explanation Generation (Step 6)
# =============================================================================

print_header "STEP 3: EXPLANATION GENERATION"
print_step "Running explanation generation..."

python "$STEP6_SCRIPT" \
    --input_file "$STEP5_OUTPUT" \
    --output_file "$STEP6_OUTPUT" \
    --explanation_prompt_path "$EXPLANATION_PROMPT" \
    --num_samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✓ Step 6 (Explanation Generation) completed successfully"
    check_file "$STEP6_OUTPUT"
    echo "✓ Output file generated: $STEP6_OUTPUT"
else
    echo "ERROR: Step 6 (Explanation Generation) failed"
    exit 1
fi

# =============================================================================
# STEP 4: Evaluation
# =============================================================================

print_header "STEP 4: PIPELINE EVALUATION"
print_step "Running evaluation..."

python "$EVALUATION_SCRIPT" \
    --input_file "$STEP6_OUTPUT" \
    --output_file "$EVALUATION_OUTPUT" \
    --device "cuda"

if [ $? -eq 0 ]; then
    echo "✓ Evaluation completed successfully"
    check_file "$EVALUATION_OUTPUT"
    echo "✓ Evaluation results saved: $EVALUATION_OUTPUT"
else
    echo "ERROR: Evaluation failed"
    exit 1
fi

# =============================================================================
# COMPLETION
# =============================================================================

print_header "PIPELINE COMPLETED SUCCESSFULLY"
echo "Pipeline execution completed at: $(date)"
echo ""
echo "Generated Files:"
echo "  - Hypotheses & Probabilities: $STEP4_OUTPUT"
echo "  - Voting Pool Results:        $STEP5_OUTPUT"
echo "  - Explanations:               $STEP6_OUTPUT"
echo "  - Evaluation Scores:          $EVALUATION_OUTPUT"
echo ""
echo "Pipeline Summary:"
echo "  - CUDA Device Used: $CUDA_VISIBLE_DEVICES"
echo "  - Environment: $CONDA_ENV_NAME"
echo "  - Samples Processed: $NUM_SAMPLES"
echo "  - Total Runtime: $(( $(date +%s) - start_time )) seconds"
echo ""

# Display quick stats if evaluation file exists
if [ -f "$EVALUATION_OUTPUT" ]; then
    echo "Quick Results Preview:"
    python -c "
import json
try:
    with open('$EVALUATION_OUTPUT', 'r') as f:
        results = json.load(f)
    print(f'  - Accuracy: {results.get(\"accuracy\", \"N/A\"):.4f}')
    print(f'  - Processed Examples: {results.get(\"processed_examples\", \"N/A\")}')
    print(f'  - BLEU-4: {results.get(\"unfiltered_scores\", {}).get(\"BLEU-4\", \"N/A\"):.4f}')
    print(f'  - BERTScore F1: {results.get(\"unfiltered_scores\", {}).get(\"BERTScore_F1\", \"N/A\"):.4f}')
except Exception as e:
    print(f'  - Could not parse results: {e}')
"
fi

echo ""
echo "🎉 All pipeline steps completed successfully!"
echo ""
