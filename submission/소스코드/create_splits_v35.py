import os
import pandas as pd

BASE = '/home/t26106/deepfake'
SRC_TRAIN = f'{BASE}/splits_v27/train_v27.csv'
SRC_VAL = f'{BASE}/splits_v27/val_v27.csv'
DST_DIR = f'{BASE}/splits_v35'
os.makedirs(DST_DIR, exist_ok=True)

def filter_ff_only(csv_path, out_path):
    df = pd.read_csv(csv_path)
    
    # Fake는 DF40, Real은 FF++만 남기기 (DF40 논문의 Protocol-1)
    fake_mask = (df['label'] == 1) & (df['dataset'].str.lower() == 'df40')
    real_mask = (df['label'] == 0) & (df['dataset'].str.lower() == 'ff')
    
    filtered_df = df[fake_mask | real_mask]
    
    # Shuffle
    filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
    filtered_df.to_csv(out_path, index=False)
    
    print(f"[{os.path.basename(out_path)}]")
    print(f"Total: {len(filtered_df)}")
    print(filtered_df.groupby(['dataset', 'label']).size().to_string())
    print("-" * 30)

if __name__ == '__main__':
    print("=== Creating v35 Splits (Protocol-1: FF++ Domain Only) ===")
    filter_ff_only(SRC_TRAIN, f'{DST_DIR}/train_v35.csv')
    filter_ff_only(SRC_VAL, f'{DST_DIR}/val_v35.csv')
