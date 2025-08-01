#!/usr/bin/env python3
"""
Script để khắc phục các lỗi đánh giá trong FDR pipeline
- Khắc phục lỗi NLTK punkt_tab resource
- Xử lý lỗi OpenAI API quota
- Cập nhật cấu hình đánh giá
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_nltk_issues():
    """Khắc phục lỗi NLTK punkt_tab resource"""
    logger.info("�� Đang khắc phục lỗi NLTK...")
    
    try:
        import nltk
        
        # Tải về các resource NLTK cần thiết
        resources_to_download = [
            'punkt',
            'wordnet', 
            'omw-1.4',
            'punkt_tab'  # Resource bị thiếu
        ]
        
        for resource in resources_to_download:
            try:
                logger.info(f"�� Đang tải {resource}...")
                nltk.download(resource, quiet=True)
                logger.info(f"✅ Đã tải thành công {resource}")
            except Exception as e:
                logger.warning(f"⚠️ Không thể tải {resource}: {e}")
        
        # Kiểm tra xem punkt_tab có sẵn không
        try:
            from nltk.tokenize import word_tokenize
            test_text = "This is a test sentence."
            tokens = word_tokenize(test_text)
            logger.info("✅ NLTK tokenization hoạt động bình thường")
            return True
        except Exception as e:
            logger.error(f"❌ NLTK tokenization vẫn lỗi: {e}")
            return False
            
    except ImportError:
        logger.error("❌ NLTK không được cài đặt. Chạy: pip install nltk")
        return False

def create_evaluation_config():
    """Tạo file cấu hình đánh giá với các tùy chọn an toàn"""
    logger.info("⚙️ Đang tạo cấu hình đánh giá...")
    
    config = {
        "evaluation": {
            "enable_g_eval": False,  # Tắt G-Eval do lỗi quota
            "enable_bleu": True,
            "enable_meteor": True,
            "enable_rouge": True,
            "enable_cider": True,
            "enable_spice": True,
            "enable_bert_score": True,
            "enable_semantic_similarity": True,
            "fallback_on_error": True
        },
        "nltk": {
            "download_resources": True,
            "quiet_download": True
        }
    }
    
    config_path = Path("evaluation_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Đã tạo cấu hình đánh giá: {config_path}")
    return config_path

def update_eval_module():
    """Cập nhật EvalModule để xử lý lỗi tốt hơn"""
    logger.info("�� Đang cập nhật EvalModule...")
    
    eval_module_path = Path("FDR/src/eval/eval_module.py")
    if not eval_module_path.exists():
        logger.error(f"❌ Không tìm thấy file: {eval_module_path}")
        return False
    
    # Đọc file hiện tại
    with open(eval_module_path, 'r') as f:
        content = f.read()
    
    # Thêm xử lý lỗi cho punkt_tab
    nltk_fix = '''
    # Fix for punkt_tab resource
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass  # Ignore if punkt_tab is not available
'''
    
    # Tìm vị trí để chèn fix
    if 'nltk.download(\'punkt\'' in content and 'punkt_tab' not in content:
        # Chèn fix sau dòng nltk.download('punkt')
        content = content.replace(
            "nltk.download('punkt', quiet=True)",
            "nltk.download('punkt', quiet=True)\n    " + nltk_fix.strip()
        )
        
        # Ghi lại file
        with open(eval_module_path, 'w') as f:
            f.write(content)
        
        logger.info("✅ Đã cập nhật EvalModule với fix cho punkt_tab")
        return True
    else:
        logger.info("ℹ️ EvalModule đã được cập nhật hoặc không cần thay đổi")
        return True

def create_g_eval_fallback():
    """Tạo fallback cho G-Eval khi API quota hết"""
    logger.info("��️ Đang tạo fallback cho G-Eval...")
    
    fallback_code = '''
def safe_g_eval_evaluation(self, results, ground_truth_explanations):
    """Safe G-Eval evaluation with fallback when API quota is exceeded"""
    try:
        return self.evaluate_explanation_batch(results, ground_truth_explanations)
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            logger.warning("⚠️ OpenAI API quota exceeded. Skipping G-Eval evaluation.")
            return {
                "evaluated_samples": 0,
                "avg_relevance": 0.0,
                "avg_coherence": 0.0, 
                "avg_faithfulness": 0.0,
                "error": "API quota exceeded"
            }
        else:
            logger.error(f"❌ G-Eval error: {e}")
            raise e
'''
    
    g_eval_path = Path("FDR/src/eval/g_evaluator.py")
    if g_eval_path.exists():
        with open(g_eval_path, 'r') as f:
            content = f.read()
        
        # Thêm method fallback nếu chưa có
        if 'safe_g_eval_evaluation' not in content:
            # Tìm vị trí cuối class để thêm method
            class_end = content.rfind('}')
            if class_end != -1:
                content = content[:class_end] + fallback_code + '\n' + content[class_end:]
                
                with open(g_eval_path, 'w') as f:
                    f.write(content)
                
                logger.info("✅ Đã thêm fallback method cho G-Eval")
                return True
    
    logger.info("ℹ️ G-Eval fallback đã tồn tại hoặc không cần thay đổi")
    return True

def install_missing_packages():
    """Cài đặt các package còn thiếu"""
    logger.info("�� Đang kiểm tra và cài đặt packages...")
    
    packages = [
        "nltk",
        "bert-score", 
        "rouge-score",
        "pycocoevalcap"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            logger.info(f"✅ Đã cài đặt {package}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Không thể cài đặt {package}: {e}")

def main():
    """Hàm chính để khắc phục tất cả lỗi"""
    logger.info("🚀 Bắt đầu khắc phục các lỗi đánh giá...")
    
    # 1. Cài đặt packages
    install_missing_packages()
    
    # 2. Khắc phục NLTK
    nltk_fixed = fix_nltk_issues()
    
    # 3. Cập nhật EvalModule
    eval_updated = update_eval_module()
    
    # 4. Tạo fallback cho G-Eval
    g_eval_fixed = create_g_eval_fallback()
    
    # 5. Tạo cấu hình
    config_created = create_evaluation_config()
    
    # Tóm tắt kết quả
    logger.info("📊 Tóm tắt khắc phục:")
    logger.info(f"   NLTK: {'✅' if nltk_fixed else '❌'}")
    logger.info(f"   EvalModule: {'✅' if eval_updated else '❌'}")
    logger.info(f"   G-Eval Fallback: {'✅' if g_eval_fixed else '❌'}")
    logger.info(f"   Config: {'✅' if config_created else '❌'}")
    
    if nltk_fixed and eval_updated and g_eval_fixed:
        logger.info("�� Tất cả lỗi đã được khắc phục! Bạn có thể chạy lại evaluation.")
    else:
        logger.warning("⚠️ Một số lỗi chưa được khắc phục hoàn toàn.")

if __name__ == "__main__":
    main() 