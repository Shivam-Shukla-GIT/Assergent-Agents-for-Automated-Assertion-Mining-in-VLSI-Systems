#!/usr/bin/env python3
"""
install_packages.py - Install all required packages with Python 3.9 compatible versions

Run this ONCE when setting up the environment:
    conda activate myenv
    python3 install_packages.py
"""

import subprocess
import sys

def run_command(cmd):
    """Run shell command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:", result.stderr)
    return result.returncode

def main():
    print("""
+----------------------------------------------------------+
¦   ASSERGENT Package Installation                 ¦
¦   Python 3.9 Compatible Versions                        ¦
+----------------------------------------------------------+
""")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ required")
        return 1
    
    packages = [
        # Core packages with Python 3.9 compatible versions
        "numpy==1.26.4",
        "requests==2.32.5",
        "scikit-learn==1.6.1",
        
        # Transformers - compatible version for Python 3.9
        "transformers==4.36.0",  # This version supports Python 3.9
        
        # Sentence transformers - compatible version
        "sentence-transformers==2.7.0",  # Compatible with transformers 4.36
        
        # Other utilities
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
    ]
    
    print("\nInstalling packages...")
    print("This may take 3-5 minutes...\n")
    
    # Install all at once
    cmd = f"pip install {' '.join(packages)}"
    returncode = run_command(cmd)
    
    if returncode != 0:
        print("\n? Installation failed!")
        return 1
    
    # Verify installation
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60)
    
    verification_script = """
import sys
print(f"Python: {sys.version}")

try:
    import numpy as np
    print(f"? NumPy: {np.__version__}")
except Exception as e:
    print(f"? NumPy: {e}")

try:
    import requests
    print(f"? Requests: {requests.__version__}")
except Exception as e:
    print(f"? Requests: {e}")

try:
    import sklearn
    print(f"? Scikit-learn: {sklearn.__version__}")
except Exception as e:
    print(f"? Scikit-learn: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("? Sentence-Transformers: Installed")
except Exception as e:
    print(f"? Sentence-Transformers: {e}")

try:
    import faiss
    print("? FAISS: Installed")
except Exception as e:
    print("? FAISS: Not installed (run: conda install -c conda-forge faiss-cpu -y)")
"""
    
    result = subprocess.run([sys.executable, "-c", verification_script], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install FAISS: conda install -c conda-forge faiss-cpu -y")
    print("2. Set API key: export GEMINI_API_KEY='your-key'")
    print("3. Run: python3 prepare_database.py")
    print("4. Run: python3 rag_assertion_generator.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
