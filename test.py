import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / "GroundingDINO"))

try:
    from groundingdino.util.inference import load_model, load_image, predict
    print("✅ GroundingDINO import thành công")
    
    config_path = Path("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    checkpoint_path = Path("GroundingDINO/weights/groundingdino_swint_ogc.pth")
    
    if config_path.exists() and checkpoint_path.exists():
        model = load_model(str(config_path), str(checkpoint_path), device="cuda:0")
        print("✅ Model load thành công")
    else:
        print("❌ File config hoặc checkpoint không tồn tại")
        
except Exception as e:
    print(f"❌ Lỗi: {e}")