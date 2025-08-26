import json
import os

def load_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading file {filepath}: {e}")
        return None

def print_complete_sample(question_id, gqa_questions, gqa_rex_converted, gqa_rex_processed):
    print(f"\n{'='*80}")
    print(f"🔍 COMPLETE SAMPLE INFORMATION - ID: {question_id}")
    print(f"{'='*80}")
    
    # Get question and answer from GQA dataset
    if question_id in gqa_questions:
        q_data = gqa_questions[question_id]
        print(f"\n📋 QUESTION & ANSWER (from GQA dataset):")
        print(f"   Question: {q_data.get('question', 'N/A')}")
        print(f"   Answer: {q_data.get('answer', 'N/A')}")
        print(f"   Full Answer: {q_data.get('fullAnswer', 'N/A')}")
        print(f"   Image ID: {q_data.get('imageId', 'N/A')}")
        
        # Additional details
        if 'types' in q_data:
            types = q_data['types']
            print(f"   Question Type: {types.get('detailed', 'N/A')}")
            print(f"   Semantic Type: {types.get('semantic', 'N/A')}")
            print(f"   Structural Type: {types.get('structural', 'N/A')}")
            
        # Show semantic operations if available
        if 'semantic' in q_data:
            print(f"   Semantic Operations: {len(q_data['semantic'])} operations")
            for i, op in enumerate(q_data['semantic'][:3]):  # Show first 3
                print(f"      {i+1}. {op.get('operation', 'N/A')}: {op.get('argument', 'N/A')}")
    else:
        print(f"\n❌ Question {question_id} not found in GQA dataset")
    
    # Get explanations from GQA_REX
    print(f"\n📝 EXPLANATIONS (from GQA_REX dataset):")
    
    if question_id in gqa_rex_converted:
        exp_data = gqa_rex_converted[question_id]
        print(f"   🔄 Converted Explanation:")
        print(f"      {exp_data}")
    else:
        print(f"   ❌ No converted explanation found")
    
    if question_id in gqa_rex_processed:
        proc_data = gqa_rex_processed[question_id]
        print(f"   ⚙️ Processed Explanation:")
        print(f"      {proc_data}")
    else:
        print(f"   ❌ No processed explanation found")

def main():
    print("🌟 COMPREHENSIVE GQA_REX SAMPLE VIEWER 🌟")
    print("="*80)
    
    # Load GQA questions dataset
    gqa_questions_path = "/mnt/VLAI_data/GQA/val_balanced_questions.json"
    print(f"\n📁 Loading GQA questions: {gqa_questions_path}")
    gqa_questions = load_json_file(gqa_questions_path)
    if gqa_questions:
        print(f"   📊 Total GQA questions: {len(gqa_questions):,}")
    else:
        print("❌ Failed to load GQA questions dataset")
        return
    
    # Load GQA_REX explanation files
    gqa_rex_dir = "/mnt/VLAI_data/GQA-REX"
    
    converted_path = os.path.join(gqa_rex_dir, "converted_explanation_val.json")
    print(f"\n📁 Loading GQA_REX converted: {converted_path}")
    gqa_rex_converted = load_json_file(converted_path)
    if gqa_rex_converted:
        print(f"   📊 Total converted explanations: {len(gqa_rex_converted):,}")
    
    processed_path = os.path.join(gqa_rex_dir, "processed_explanation_val.json")
    print(f"\n📁 Loading GQA_REX processed: {processed_path}")
    gqa_rex_processed = load_json_file(processed_path)
    if gqa_rex_processed:
        print(f"   📊 Total processed explanations: {len(gqa_rex_processed):,}")
    
    # Show complete sample information
    target_sample = "05515938"
    print_complete_sample(target_sample, gqa_questions, gqa_rex_converted or {}, gqa_rex_processed or {})
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE SAMPLE VIEWING COMPLETED!")
    print("📌 This sample shows the full pipeline data including:")
    print("   - Original question and answer from GQA dataset")
    print("   - Question type and semantic operations")
    print("   - Explanations from GQA_REX (both converted and processed)")

if __name__ == "__main__":
    main()
