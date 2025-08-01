#!/usr/bin/env python3
# FDR/install_deps.py - Install dependencies for Auto Save Pipeline
import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install required dependencies for Auto Save Pipeline"""
    print("🔧 Installing dependencies for Auto Save Pipeline...")
    
    # Required packages for auto save pipeline
    packages = [
        "matplotlib",
        "Pillow",  # PIL
        "tqdm",
        "pyyaml"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation summary:")
    print(f"  Successfully installed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("✅ All dependencies installed successfully!")
        print("\n🚀 You can now run the Auto Save Pipeline:")
        print("  python quick_run.py")
    else:
        print("⚠️ Some packages failed to install. Please install manually:")
        for package in packages:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main() 