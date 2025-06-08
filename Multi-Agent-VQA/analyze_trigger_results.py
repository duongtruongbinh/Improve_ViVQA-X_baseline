#!/usr/bin/env python3
import re

def analyze_trigger_issues():
    """Phân tích các vấn đề trigger từ log output"""
    
    print("🔍 PHÂN TÍCH VẤN ĐỀ TRIGGER TỪ LOG OUTPUT")
    print("="*60)
    
    # Cases triggered từ log
    triggered_cases = [
        {
            "question": "Where is this market?",
            "baseline": "street", 
            "final": "yes",
            "target": "outside",
            "trigger": "location_trigger",
            "improved": False
        },
        {
            "question": "Is the man putting the pizza into or taking it out of the oven?", 
            "baseline": "into",
            "final": "yes", 
            "target": "in",
            "trigger": "complex_visual, low_confidence",
            "improved": False
        },
        {
            "question": "Does it look as if the child is having the adult beverage...",
            "baseline": "No",
            "final": "No",
            "target": "no", 
            "trigger": "complex_visual, low_confidence",
            "improved": False  # Same result
        },
        {
            "question": "What species is the stuffed animal behind the cat?",
            "baseline": "There is no stuffed animal in the image.",
            "final": "No stuffed animal",
            "target": "no stuffed animal",
            "trigger": "complex_visual",
            "improved": True  # Better normalization
        },
        {
            "question": "What kind of icing is on the cake?", 
            "baseline": "Chocolate",
            "final": "Chocolate",
            "target": "chocolate",
            "trigger": "complex_visual",
            "improved": False  # No improvement needed
        },
        {
            "question": "What kind of soup is this?",
            "baseline": "vegetable", 
            "final": "vegetable",
            "target": "vegetable",
            "trigger": "complex_visual", 
            "improved": False  # No improvement needed
        },
        {
            "question": "Where is this market?",
            "baseline": "Market",
            "final": "[Market]", 
            "target": "outside",
            "trigger": "location_trigger",
            "improved": False  # Made worse
        },
        {
            "question": "How many items do you see?",
            "baseline": "3",
            "final": "2", 
            "target": "3", 
            "trigger": "counting_verification",
            "improved": False  # Made worse
        },
        {
            "question": "How many kites are in the sky?",
            "baseline": "10",
            "final": "9",
            "target": "12",
            "trigger": "counting_verification", 
            "improved": False  # Both wrong
        },
        {
            "question": "How many bananas are there?",
            "baseline": "10", 
            "final": "29",
            "target": "30",
            "trigger": "counting_verification",
            "improved": True  # Closer to target
        }
    ]
    
    print("📊 TRIGGER PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    total_triggered = len(triggered_cases)
    improved_count = sum(1 for case in triggered_cases if case["improved"])
    no_change_count = sum(1 for case in triggered_cases if not case["improved"] and case["baseline"].lower().strip() == case["final"].lower().strip())
    made_worse_count = total_triggered - improved_count - no_change_count
    
    print(f"Total Triggered Cases: {total_triggered}")
    print(f"✅ Improved: {improved_count} ({improved_count/total_triggered*100:.1f}%)")
    print(f"➡️  No Change: {no_change_count} ({no_change_count/total_triggered*100:.1f}%)")  
    print(f"❌ Made Worse: {made_worse_count} ({made_worse_count/total_triggered*100:.1f}%)")
    
    print("\n🚨 MAIN ISSUES IDENTIFIED:")
    print("-" * 40)
    
    # Analyze by trigger type
    trigger_types = {}
    for case in triggered_cases:
        triggers = case["trigger"].split(", ")
        for trigger in triggers:
            if trigger not in trigger_types:
                trigger_types[trigger] = {"total": 0, "improved": 0, "worse": 0}
            trigger_types[trigger]["total"] += 1
            if case["improved"]:
                trigger_types[trigger]["improved"] += 1
            elif case["baseline"].lower().strip() != case["final"].lower().strip():
                trigger_types[trigger]["worse"] += 1
    
    for trigger, stats in trigger_types.items():
        improvement_rate = stats["improved"] / stats["total"] * 100
        print(f"🔹 {trigger}: {improvement_rate:.1f}% improvement rate ({stats['improved']}/{stats['total']})")
        if stats["worse"] > 0:
            print(f"   ⚠️  {stats['worse']} cases made worse")
    
    print("\n🎯 SPECIFIC PROBLEMS:")
    print("-" * 40)
    
    print("1. 🔄 **ANSWER FORMAT ISSUES**:")
    print("   - '[Reattempted Answer] X' format not cleaned properly")
    print("   - '[Market]' instead of clean answer")
    print("   - Need better post-processing")
    
    print("\n2. 📍 **LOCATION TRIGGER OVER-AGGRESSIVE**:")
    print("   - 'street' is actually reasonable for 'Where is this market?'")
    print("   - Should not trigger unless answer is truly insufficient")
    
    print("\n3. 🔢 **COUNTING VERIFICATION ISSUES**:")
    print("   - CLIP-Count sometimes less accurate than VLM baseline")
    print("   - Threshold of 3 too low, triggering easy cases")
    
    print("\n4. 🎨 **ATTRIBUTE TRIGGERS UNNECESSARY**:")
    print("   - 'Chocolate' is perfect answer for 'What kind of icing?'")
    print("   - Triggering when no improvement needed")
    
    print("\n5. 📝 **COMPLEX VISUAL OVER-TRIGGERING**:")
    print("   - Long questions with adequate short answers shouldn't trigger")
    print("   - 'No' is perfect for yes/no questions regardless of length")
    
    return trigger_types

def recommend_fixes():
    print("\n🔧 RECOMMENDED FIXES:")
    print("="*60)
    
    print("1. **INCREASE TRIGGER SELECTIVITY**:")
    print("   - Raise counting threshold back to 5-6")
    print("   - More restrictive location triggers")
    print("   - Skip attribute triggers if answer already contains relevant info")
    
    print("\n2. **IMPROVE ANSWER POST-PROCESSING**:")
    print("   - Better cleanup of '[Reattempted Answer]' format")
    print("   - Normalize bracket notation")
    print("   - Extract core answer from verbose responses")
    
    print("\n3. **SMARTER OVERRIDE CONDITIONS**:")
    print("   - Don't trigger if baseline answer quality is already good")
    print("   - Check answer relevance before triggering")
    print("   - Consider question difficulty vs answer adequacy")
    
    print("\n4. **COUNTING STRATEGY REVISION**:")
    print("   - Only use CLIP-Count for truly difficult counting (>10 objects)")
    print("   - Fall back to baseline for simple counting")
    print("   - Better object detection thresholds")

if __name__ == "__main__":
    trigger_stats = analyze_trigger_issues()
    recommend_fixes() 