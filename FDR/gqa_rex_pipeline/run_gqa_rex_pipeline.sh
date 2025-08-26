#!/bin/bash
#
# GQA-REX Pipeline Execution Script
# Based on CLEVR-X execution pattern
#

echo "🎯 GQA-REX Multi-Agent Pipeline"
echo "==============================="

# Check if servers are running
echo "🔍 Checking server status..."

VL_STATUS="❌ Not responding"
LLM_STATUS="❌ Not responding"

if curl -s http://localhost:9100/v1/models > /dev/null; then
    VL_STATUS="✅ Running"
fi

if curl -s http://localhost:9200/v1/models > /dev/null; then
    LLM_STATUS="✅ Running"
fi

echo "   • VL Server (Port 9100): $VL_STATUS"
echo "   • LLM Server (Port 9200): $LLM_STATUS"

if [[ "$VL_STATUS" == "❌ Not responding" ]] || [[ "$LLM_STATUS" == "❌ Not responding" ]]; then
    echo ""
    echo "❌ Servers not ready! Please run: bash run_gqa_rex_servers.sh"
    exit 1
fi

echo ""
echo "📊 Dataset Information:"
echo "   • GQA Dataset: Visual questions with scene graphs"
echo "   • GQA-REX Dataset: Natural language explanations"
echo "   • Validation Split: ~127,900 linked samples"
echo ""

# Default parameters - GQA-REX Validation Set
SPLIT="val"  # Use validation set for evaluation
NUM_SAMPLES=100  # Process 100 samples by default
OUTPUT_DIR="results_gqa_rex_val"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash run_gqa_rex_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --split SPLIT          Dataset split (val/train/test, default: val)"
            echo "  --num_samples NUM      Number of samples to process (default: 100)"
            echo "  --output_dir DIR       Output directory (default: results_gqa_rex)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  bash run_gqa_rex_pipeline.sh --split val --num_samples 50"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🎛️  Pipeline Configuration:"
echo "   • Split: $SPLIT"
echo "   • Samples: $NUM_SAMPLES"
echo "   • Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting GQA-REX Pipeline..."
echo "   Time: $(date)"
echo ""

# Change to FDR directory
cd /home/huytd/multi-agent/multi-agent/FDR

# Run the GQA-REX pipeline
python gqa_rex_pipeline/main_gqa_rex.py \
    --dataset "gqa_rex_${SPLIT}" \
    --samples "$NUM_SAMPLES" \
    --backend vllm \
    --evaluate

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ GQA-REX Pipeline completed successfully!"
    echo "   Results saved to: $OUTPUT_DIR"
    echo "   Time: $(date)"
    
    # Show output statistics if results exist
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo ""
        echo "📈 Results Summary:"
        python -c "
import json
try:
    with open('$OUTPUT_DIR/results.json', 'r') as f:
        results = json.load(f)
    print(f'   • Total processed: {len(results)}')
    if results:
        correct = sum(1 for r in results if r.get('correct', False))
        print(f'   • Accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)')
except:
    pass
"
    fi
else
    echo "❌ Pipeline failed with exit code: $EXIT_CODE"
fi

echo ""
echo "📝 Log files and detailed results are in: $OUTPUT_DIR/"
