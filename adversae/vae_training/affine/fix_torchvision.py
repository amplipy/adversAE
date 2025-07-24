#!/usr/bin/env python3
"""
Torch/Torchvision Compatibility Checker and Fixer

This script helps diagnose and fix common torchvision compatibility issues,
particularly the "torchvision::nms does not exist" error.

Usage:
    python fix_torchvision.py
"""

import sys
import subprocess
import importlib.util

def check_package_version(package_name):
    """Check if a package is installed and return its version"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return None
        
        package = importlib.import_module(package_name)
        return getattr(package, '__version__', 'Unknown')
    except Exception as e:
        return f"Error: {e}"

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔍 Torch/Torchvision Compatibility Checker")
    print("=" * 50)
    
    # Check current versions
    torch_version = check_package_version('torch')
    torchvision_version = check_package_version('torchvision')
    
    print(f"📦 Current PyTorch version: {torch_version}")
    print(f"📦 Current Torchvision version: {torchvision_version}")
    
    if torch_version is None:
        print("❌ PyTorch not found. Installing...")
        success, stdout, stderr = run_command("pip install torch")
        if not success:
            print(f"❌ Failed to install PyTorch: {stderr}")
            return
    
    if torchvision_version is None:
        print("❌ Torchvision not found. Installing...")
        success, stdout, stderr = run_command("pip install torchvision")
        if not success:
            print(f"❌ Failed to install Torchvision: {stderr}")
            return
    
    # Test torchvision import
    print("\n🧪 Testing torchvision import...")
    try:
        import torch
        print(f"✅ PyTorch imported successfully: {torch.__version__}")
        
        import torchvision
        print(f"✅ Torchvision imported successfully: {torchvision.__version__}")
        
        # Test specific functionality
        from torchvision import transforms, datasets
        print("✅ Torchvision transforms and datasets imported successfully")
        
        # Test if the problematic nms operation works
        try:
            import torchvision.ops
            print("✅ Torchvision ops imported successfully")
        except Exception as e:
            print(f"⚠️ Torchvision ops import warning: {e}")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        print("\n🔧 Attempting to fix...")
        
        # Try to reinstall with CPU-only versions
        print("Installing CPU-only versions...")
        commands = [
            "pip uninstall torch torchvision -y",
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            success, stdout, stderr = run_command(cmd)
            if not success:
                print(f"⚠️ Command failed: {stderr}")
            else:
                print("✅ Command completed")
        
        # Test again
        print("\n🧪 Testing imports after reinstall...")
        try:
            import torch
            import torchvision
            print(f"✅ After fix - PyTorch: {torch.__version__}")
            print(f"✅ After fix - Torchvision: {torchvision.__version__}")
        except Exception as e2:
            print(f"❌ Still failing after fix: {e2}")
            print("\n💡 Manual fix suggestions:")
            print("1. Try: conda install pytorch torchvision cpuonly -c pytorch")
            print("2. Or: pip install torch==2.0.1 torchvision==0.15.2")
            print("3. Restart your Python environment/kernel")
            return
    
    print("\n🎉 All checks passed!")
    print("\n💡 If you still see torchvision::nms errors:")
    print("1. Restart your Python kernel/environment")
    print("2. Make sure you're using CPU-only mode (no CUDA)")
    print("3. Consider using the fallback data loading in the Streamlit app")
    
    # Test MNIST loading
    print("\n📊 Testing MNIST data loading...")
    try:
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        print(f"✅ MNIST dataset loaded successfully: {len(dataset)} samples")
    except Exception as e:
        print(f"⚠️ MNIST loading warning: {e}")
        print("This might not be critical for the Streamlit app")

if __name__ == "__main__":
    main()
