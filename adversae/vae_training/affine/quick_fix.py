#!/usr/bin/env python3
"""
Quick fix for torchvision::nms compatibility issue
"""
import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command: {e}")
        return False

def main():
    print("🔧 Quick fix for torchvision compatibility...")
    
    # Try to uninstall and reinstall compatible versions
    print("\n1. Uninstalling current versions...")
    run_command("pip uninstall torch torchvision -y")
    
    print("\n2. Installing compatible versions...")
    success = run_command("pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu")
    
    if success:
        print("\n✅ Installation complete!")
        print("Please restart your Jupyter kernel now.")
    else:
        print("\n❌ Installation failed. Trying alternative...")
        run_command("pip install torch torchvision --force-reinstall")
    
    print("\n3. Testing imports...")
    try:
        import torch
        import torchvision
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ Torchvision {torchvision.__version__}")
        
        # Test the problematic import
        from torchvision import datasets, transforms
        print("✅ Torchvision imports working!")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("You may need to restart Python completely.")

if __name__ == "__main__":
    main()
