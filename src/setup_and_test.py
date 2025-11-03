import subprocess
import sys
import os
from pathlib import Path
import time

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_python_version():
    """Check Python version"""
    print_section("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"\n  Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"  Location: {sys.executable}")
    
    if version.major >= 3 and version.minor >= 8:
        print("  ✅ Python version OK (3.8+)")
        return True
    else:
        print("  ❌ Python 3.8+ required")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print_section("CHECKING DEPENDENCIES")
    
    required_packages = {
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'pydantic': 'Pydantic',
        'requests': 'Requests'
    }
    
    print("\n📦 Required Packages:")
    missing = []
    
    for package, name in required_packages.items():
        if check_package(package):
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n  ⚠️  Missing packages: {', '.join(missing)}")
        print(f"\n  To install:")
        print(f"    pip install {' '.join(missing)}")
        return False
    else:
        print("\n  ✅ All dependencies installed!")
        return True

def check_directory_structure():
    """Check if directory structure is correct"""
    print_section("CHECKING DIRECTORY STRUCTURE")
    
    required_dirs = ['src', 'data', 'data/raw', 'data/processed', 'data/models', 'outputs']
    required_files = ['src/__init__.py', 'src/config.py', 'src/data_preprocessing.py', 
                     'src/models.py', 'src/trainer.py', 'src/evaluation.py']
    
    all_good = True
    
    print("\n📁 Directories:")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - MISSING")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"     ✅ Created {dir_path}/")
    
    print("\n📄 Required Files:")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_data():
    """Check if data files exist"""
    print_section("CHECKING DATA")
    
    raw_data_paths = [
        Path('data/raw/amazon_review.csv'),
        Path('data/raw/amazon_reviews.csv')
    ]
    
    for raw_data in raw_data_paths:
        if raw_data.exists():
            size_mb = raw_data.stat().st_size / (1024 * 1024)
            print(f"\n  ✅ Raw data found: {raw_data}")
            print(f"     Size: {size_mb:.2f} MB")
            return True
    
    print(f"\n  ⚠️  Raw data not found in data/raw/")
    return False

def generate_summary():
    """Generate summary report"""
    print_section("SYSTEM STATUS SUMMARY")
    
    checks = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        "Directory structure": check_directory_structure(),
        "Raw data": check_data()
    }
    
    print("\n" + "=" * 80)
    print("  FINAL STATUS")
    print("=" * 80 + "\n")
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {check}")
    
    print(f"\n  Score: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n  🎉 SYSTEM READY!")
        print("\n  Next steps:")
        print("    python main.py --preprocess")
        print("    python main.py --train --model ncf")
    else:
        print("\n  ⚠️  Please fix the issues above")
    
    return passed, total

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("  🛍️  AMAZON RECOMMENDATION SYSTEM")
    print("       Setup & Testing Script")
    print("=" * 80)
    print(f"\n  Current directory: {os.getcwd()}")
    
    passed, total = generate_summary()
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)