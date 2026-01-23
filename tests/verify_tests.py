#!/usr/bin/env python3
"""
Verify test environment and dependencies
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_dependency(package_name):
    """Check if a Python package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def verify_test_environment():
    """Verify the test environment is properly set up"""
    
    print("🔍 Verifying Qwen3-TTS Test Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = [
        'pytest',
        'requests', 
        'numpy',
        'soundfile'
    ]
    
    print("\n📦 Checking required packages:")
    all_packages_ok = True
    
    for package in required_packages:
        if check_dependency(package):
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - Missing!")
            all_packages_ok = False
    
    # Check test directory structure
    print("\n📁 Checking test directory structure:")
    test_dir = Path("tests")
    
    required_files = [
        "tests/__init__.py",
        "tests/conftest.py", 
        "tests/test_health_api.py",
        "tests/test_tts_api.py",
        "tests/pytest.ini",
        "tests/README.md"
    ]
    
    structure_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            structure_ok = False
    
    # Check run script
    run_script = Path("run_tests.sh")
    if run_script.exists() and os.access(run_script, os.X_OK):
        print("✅ run_tests.sh (executable)")
    else:
        print("❌ run_tests.sh - Missing or not executable!")
        structure_ok = False
    
    # Check pytest configuration
    print("\n⚙️  Checking pytest configuration:")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Pytest is working")
            print(f"   {result.stdout.strip()}")
        else:
            print("❌ Pytest configuration issue")
            print(result.stderr)
            all_packages_ok = False
    except Exception as e:
        print(f"❌ Error checking pytest: {e}")
        all_packages_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_packages_ok and structure_ok:
        print("🎉 Test environment is ready!")
        print("\nNext steps:")
        print("1. Start the Qwen3-TTS API server")
        print("2. Run tests: ./run_tests.sh -t smoke")
        return True
    else:
        print("❌ Test environment has issues")
        
        if not all_packages_ok:
            print("\nTo install missing packages:")
            print("pip install -r requirements.txt")
        
        if not structure_ok:
            print("\nMissing test files - please check the test directory structure")
        
        return False

if __name__ == "__main__":
    success = verify_test_environment()
    sys.exit(0 if success else 1)
