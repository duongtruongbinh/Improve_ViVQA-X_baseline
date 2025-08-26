#!/usr/bin/env python3
"""
GQA-REX Dataset Sample Viewer
View 1 sample from each GQA-REX validation file
"""

import json
import os

def load_json_file(file_path):
    """Load and return JSON data from file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def print_sample(data, file_name):
    """Print 1 sample from the data"""
    print("\n" + "="*80)
    print(f"📁 FILE: {file_name}")
    print("="*80)
    print(f"📊 Total samples: {len(data):,}")
    print("🔍 Showing 1 sample:")
    print("="*80)
    
    # Get first item
    first_key = next(iter(data))
    first_value = data[first_key]
    
    print(f"\n🔸 Question ID: {first_key}")
    print(f"🔸 Explanation: {first_value}")
    
    print("\n" + "="*80)
    print("✅ Sample display completed")
    print("="*80)

def main():
    """Main function"""
    print("🚀 GQA-REX Dataset Sample Viewer")
    print("Viewing 1 sample from each validation file\n")
    
    files = {
        'converted_explanation_val.json': '/mnt/VLAI_data/GQA-REX/converted_explanation_val.json',
        'processed_explanation_val.json': '/mnt/VLAI_data/GQA-REX/processed_explanation_val.json'
    }
    
    for file_name, file_path in files.items():
        try:
            print(f"\n🔄 Loading {file_name}...")
            data = load_json_file(file_path)
            print(f"✅ Successfully loaded!")
            
            if isinstance(data, dict) and data:
                print_sample(data, file_name)
            else:
                print(f"❌ Invalid data format in {file_name}")
                
        except Exception as e:
            print(f"❌ Error loading {file_name}: {str(e)}")
    
    print(f"\n{'🎯 ' + '='*76 + ' 🎯'}")
    print("🏁 All files processed!")
    print(f"{'🎯 ' + '='*76 + ' 🎯'}")

if __name__ == "__main__":
    main()
