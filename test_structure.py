#!/usr/bin/env python3

import os
import sys

def test_project_structure():
    """Test that all required files and directories exist"""
    
    print("🔍 Testing Project Structure...")
    print("-" * 50)
    
    required_dirs = [
        'src',
        'src/agents',
        'src/tools',
        'src/chains',
        'src/models',
        'src/utils',
        'tests',
        'notebooks'
    ]
    
    required_files = [
        'requirements.txt',
        '.env',
        '.gitignore',
        'README.md',
        'src/__init__.py',
        'src/config.py',
        'src/main.py',
        'src/agents/__init__.py',
        'src/agents/base_agent.py',
        'src/tools/__init__.py',
        'src/tools/data_collection.py',
        'src/tools/feature_engineering.py',
        'src/tools/risk_assessment.py',
        'src/models/__init__.py',
        'src/models/schemas.py',
        'src/models/validators.py',
    ]
    
    errors = []
    
    print("📂 Checking directories...")
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing!")
            errors.append(f"Missing directory: {dir_path}")
    
    print("\n📄 Checking files...")
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} - Missing!")
            errors.append(f"Missing file: {file_path}")
    
    print("\n" + "-" * 50)
    
    if errors:
        print(f"❌ Found {len(errors)} issues:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ All required files and directories are present!")
        return True

def test_imports():
    """Test that basic Python imports work"""
    
    print("\n📦 Testing Python imports...")
    print("-" * 50)
    
    basic_imports = [
        ('datetime', 'datetime'),
        ('json', None),
        ('os', None),
        ('sys', None),
        ('typing', 'Dict'),
        ('enum', 'Enum'),
        ('dataclasses', 'dataclass'),
    ]
    
    errors = []
    
    for module_name, attr in basic_imports:
        try:
            if attr:
                exec(f"from {module_name} import {attr}")
                print(f"  ✅ from {module_name} import {attr}")
            else:
                exec(f"import {module_name}")
                print(f"  ✅ import {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name} - {e}")
            errors.append(f"Import error: {module_name}")
    
    print("\n" + "-" * 50)
    
    if errors:
        print(f"❌ Found {len(errors)} import issues")
        return False
    else:
        print("✅ All basic imports work!")
        return True

def main():
    print("=" * 50)
    print("🚀 Crypto Wallet Recommendation Agent")
    print("   Project Structure Test")
    print("=" * 50)
    
    structure_ok = test_project_structure()
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"  Structure: {'✅ Pass' if structure_ok else '❌ Fail'}")
    print(f"  Imports:   {'✅ Pass' if imports_ok else '❌ Fail'}")
    
    if structure_ok and imports_ok:
        print("\n✅ Project structure is ready!")
        print("\n📝 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Configure API keys in .env file")
        print("  3. Run: python src/main.py test")
        return 0
    else:
        print("\n❌ Project has issues that need to be fixed")
        return 1

if __name__ == "__main__":
    sys.exit(main())