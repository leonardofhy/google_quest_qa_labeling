import pandas as pd
import numpy as np
import argparse
import sys
import os

def verify_submission(submission_path, sample_path):
    print(f"Verifying submission file: {submission_path}")
    
    # 1. Check existence
    if not os.path.exists(submission_path):
        print(f"[ERROR] Submission file not found: {submission_path}")
        return False

    try:
        df_sub = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return False

    # Load sample for comparison
    if not os.path.exists(sample_path):
         print(f"[WARNING] Sample submission not found at {sample_path}. Skipping strict ID and column comparison.")
         df_sample = None
         expected_cols = None
    else:
        df_sample = pd.read_csv(sample_path)
        expected_cols = list(df_sample.columns)

    # 2. Check Columns
    if expected_cols:
        if list(df_sub.columns) != expected_cols:
            print(f"[ERROR] Column mismatch!")
            
            # Check for missing or extra
            missing = set(expected_cols) - set(df_sub.columns)
            extra = set(df_sub.columns) - set(expected_cols)
            if missing: print(f"  - Missing columns: {missing}")
            if extra: print(f"  - Extra columns: {extra}")
            
            # Check order
            if not missing and not extra:
                print("  - Columns match but order is different. Reordering...")
                df_sub = df_sub[expected_cols]
            else:
                return False
    else:
        # Minimal check if sample not found: qa_id must exist
        if 'qa_id' not in df_sub.columns:
            print("[ERROR] 'qa_id' column missing.")
            return False
            
    # 3. Check Data Types
    if not pd.api.types.is_integer_dtype(df_sub['qa_id']):
        print(f"[ERROR] 'qa_id' column should be integer. Found: {df_sub['qa_id'].dtype}")
        return False
        
    # Check target columns are numeric and specifically float
    target_cols = [c for c in df_sub.columns if c != 'qa_id']
    for col in target_cols:
        if not pd.api.types.is_float_dtype(df_sub[col]):
             # Try to convert if it's numeric but maybe int (e.g. all 0s and 1s)
            if pd.api.types.is_numeric_dtype(df_sub[col]):
                print(f"[WARNING] Column '{col}' is numeric but not float (found {df_sub[col].dtype}). Casting to float.")
                df_sub[col] = df_sub[col].astype(float)
            else:
                print(f"[ERROR] Column '{col}' should be float. Found: {df_sub[col].dtype}")
                return False

    # 4. Check Value Range [0, 1]
    min_val = df_sub[target_cols].min().min()
    max_val = df_sub[target_cols].max().max()
    
    # Allow small epsilon for floating point errors if necessary
    if min_val < -1e-9 or max_val > 1.0 + 1e-9:
        print(f"[ERROR] Values out of range [0, 1]. Min: {min_val}, Max: {max_val}")
        # Identify columns with issues
        for col in target_cols:
            c_min = df_sub[col].min()
            c_max = df_sub[col].max()
            if c_min < -1e-9 or c_max > 1.0 + 1e-9:
                print(f"  - {col}: range [{c_min}, {c_max}]")
        return False

    # 5. Check for NaNs
    if df_sub.isnull().any().any():
        print("[ERROR] Found NaN values in submission.")
        print(df_sub.isnull().sum()[df_sub.isnull().sum() > 0])
        return False
        
    # 6. Check for Infinity
    if np.isinf(df_sub[target_cols].values).any():
        print("[ERROR] Found Infinite values in submission.")
        return False

    # 7. Check qa_id uniqueness and matching with sample
    if df_sub['qa_id'].nunique() != len(df_sub):
        print("[ERROR] 'qa_id' is not unique.")
        return False

    if df_sample is not None:
        # Strict Set Check
        sub_ids = set(df_sub['qa_id'])
        sample_ids = set(df_sample['qa_id'])
        
        if sub_ids != sample_ids:
            print("[ERROR] Mismatch in 'qa_id's between submission and sample.")
            missing_ids = sample_ids - sub_ids
            extra_ids = sub_ids - sample_ids
            if missing_ids: print(f"  - Missing IDs (in sample but not submission): {len(missing_ids)} example: {list(missing_ids)[:5]}")
            if extra_ids: print(f"  - Extra IDs (in submission but not sample): {len(extra_ids)} example: {list(extra_ids)[:5]}")
            return False
            
        # Strict Order Check
        if not df_sub['qa_id'].equals(df_sample['qa_id']):
            print("[WARNING] 'qa_id' order does not match sample submission. Reordering to match sample...")
            df_sub = df_sub.set_index('qa_id').reindex(df_sample['qa_id']).reset_index()
            
    # 8. Sanity Check: Constant Predictions
    # Warn if a column has 0 variance (unless it's intended, but usually suspicious for regression)
    constant_cols = []
    for col in target_cols:
        if df_sub[col].std() == 0:
            constant_cols.append(col)
            
    if constant_cols:
        print(f"[WARNING] The following columns have constant values (std dev = 0):")
        for c in constant_cols:
            print(f"  - {c}: value = {df_sub[c].iloc[0]}")
        print("  (This might be expected for some post-processing or weak targets, but verify!)")

    print("[SUCCESS] Submission file passed all verification checks!")
    print(f"  - Shape: {df_sub.shape}")
    print(f"  - Columns: {len(df_sub.columns)}")
    print(f"  - Value Range: [{min_val:.5f}, {max_val:.5f}]")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Kaggle Submission CSV")
    parser.add_argument("submission_csv", help="Path to the submission CSV file")
    parser.add_argument("--sample_csv", default="/home/leonardo298/Workspace/google_quest_qa_labeling/data/sample_submission.csv", help="Path to sample submission for schema validation")
    
    args = parser.parse_args()
    
    success = verify_submission(args.submission_csv, args.sample_csv)
    sys.exit(0 if success else 1)
