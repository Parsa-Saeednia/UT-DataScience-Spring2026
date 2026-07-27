import subprocess
import sys

def run_pipeline():
    print("Starting Data Science Pipeline: ")
    
    print("\nStep 1: Loading Data: ")
    result = subprocess.run(["python", "scripts/load_data.py"])
    if result.returncode != 0:
        sys.exit(1)
        
    print("\nStep 2: Preprocessing: ")
    result = subprocess.run(["python", "scripts/preprocess.py"])
    if result.returncode != 0:
        sys.exit(1)
        
    print("\nStep 3: Feature Engineering: ")
    result = subprocess.run(["python", "scripts/feature_engineering.py"])
    if result.returncode != 0:
        sys.exit(1)
        
    print("\nStep 4: Model Training: ")
    result = subprocess.run(["python", "scripts/train.py"])
    if result.returncode != 0:
        sys.exit(1)
        
    print("\nStep 5: Model Evaluation: ")
    result = subprocess.run(["python", "scripts/evaluate.py"])
    if result.returncode != 0:
        sys.exit(1)
        
    print("\nPipeline Execution Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()