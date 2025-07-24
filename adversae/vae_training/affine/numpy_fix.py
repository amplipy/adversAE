#!/usr/bin/env python3
"""
Definitive fix for numpy 2.x compatibility issues
This script will downgrade numpy to 1.x and reinstall all compatible packages
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command with better error handling"""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}")
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
        
        return result.returncode == 0
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"🚨 {description} - EXCEPTION: {e}")
        return False

def main():
    print("🛠️ NUMPY 2.x COMPATIBILITY FIX")
    print("=" * 50)
    
    print("\n📋 PLAN:")
    print("1. Uninstall conflicting packages")
    print("2. Install numpy 1.24.x")
    print("3. Install compatible matplotlib, pytorch, etc.")
    print("4. Test all imports")
    
    # Step 1: Uninstall problematic packages
    packages_to_remove = ["numpy", "matplotlib", "scipy", "pandas", "seaborn"]
    for package in packages_to_remove:
        run_command(f"pip uninstall {package} -y", f"Uninstall {package}")
    
    # Step 2: Install compatible numpy first
    run_command(
        "pip install 'numpy<2.0' --no-deps", 
        "Install numpy 1.x"
    )
    
    # Step 3: Install compatible packages
    compatible_packages = [
        "matplotlib>=3.5.0,<4.0",
        "scipy>=1.7.0,<2.0", 
        "pandas>=1.3.0,<3.0",
        "seaborn>=0.11.0",
        "pillow>=8.3.0"
    ]
    
    for package in compatible_packages:
        run_command(f"pip install '{package}'", f"Install {package}")
    
    # Step 4: Reinstall PyTorch with compatible numpy
    run_command(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "Reinstall PyTorch with compatible dependencies"
    )
    
    # Step 5: Test imports
    print("\n🧪 TESTING IMPORTS...")
    
    test_imports = [
        ("numpy", "import numpy as np; print(f'Numpy: {np.__version__}')"),
        ("matplotlib", "import matplotlib.pyplot as plt; print('Matplotlib: OK')"),
        ("torch", "import torch; print(f'PyTorch: {torch.__version__}')"),
        ("torchvision", "from torchvision import transforms; print('Torchvision: OK')"),
        ("scipy", "import scipy; print(f'Scipy: {scipy.__version__}')"),
        ("pandas", "import pandas as pd; print(f'Pandas: {pd.__version__}')"),
        ("seaborn", "import seaborn as sns; print('Seaborn: OK')")
    ]
    
    working = []
    failed = []
    
    for name, test_code in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                working.append(name)
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                failed.append(name)
                print(f"❌ {name}: {result.stderr.strip()[:100]}")
        except Exception as e:
            failed.append(name)
            print(f"🚨 {name}: {e}")
    
    # Final report
    print("\n" + "=" * 50)
    print("📊 FINAL REPORT")
    print(f"✅ Working packages ({len(working)}): {', '.join(working)}")
    if failed:
        print(f"❌ Failed packages ({len(failed)}): {', '.join(failed)}")
    
    if len(working) >= 4:  # numpy, matplotlib, torch, torchvision
        print("\n🎉 SUCCESS! Core packages are working.")
        print("🔄 Please restart your Jupyter kernel completely.")
    else:
        print("\n⚠️ Some issues remain. You may need to:")
        print("1. Restart your terminal/environment completely")
        print("2. Consider creating a fresh conda environment")
        print("3. Contact your system administrator if on a managed system")

if __name__ == "__main__":
    main()
