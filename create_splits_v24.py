"""
create_splits_v24.py
v20 학습 CSV (splits_v11_v2) + SNS real 데이터 병합 → splits_v24/
SNS 영상은 video_name 기준 80/20 train/val 분할
"""
import os
import pandas as pd
import numpy as np

V20_SPLITS = '/home/t26106/deepfake/splits_v11_v2'
SNS_CSV    = '/home/t26106/deepfake/splits/sns_real_frames.csv'
OUT_DIR    = '/home/t26106/deepfake/splits_v24'
SEED       = 42

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# 1. SNS real 분할
sns = pd.read_csv(SNS_CSV)
videos = sns['video_name'].unique()
rng.shuffle(videos)
n_val = max(1, int(len(videos) * 0.2))
val_videos = set(videos[:n_val])
train_videos = set(videos[n_val:])

sns_train = sns[sns['video_name'].isin(train_videos)].copy()
sns_val = sns[sns['video_name'].isin(val_videos)].copy()
sns_train['split'] = 'train'
sns_val['split'] = 'val'

print(f'SNS 영상 분할: train={len(train_videos)}개 ({len(sns_train)}프레임) / val={len(val_videos)}개 ({len(sns_val)}프레임)')

# 2. v20 학습 CSV (splits_v11_v2) 로드
v20_train = pd.read_csv(os.path.join(V20_SPLITS, 'train_v11.csv'))
v20_val = pd.read_csv(os.path.join(V20_SPLITS, 'val_v11.csv'))
print(f'v20 train: {len(v20_train)}프레임, val: {len(v20_val)}프레임')

# 3. 컬럼 정렬 확인
def align(df, ref_cols):
    for c in ref_cols:
        if c not in df.columns:
            df[c] = ''
    return df[ref_cols]

ref_cols = list(v20_train.columns)
sns_train = align(sns_train, ref_cols)
sns_val = align(sns_val, ref_cols)

# 4. 병합
train_v24 = pd.concat([v20_train, sns_train], ignore_index=True)
val_v24   = pd.concat([v20_val, sns_val], ignore_index=True)

# 5. 저장
train_v24.to_csv(os.path.join(OUT_DIR, 'train_v24.csv'), index=False)
val_v24.to_csv(os.path.join(OUT_DIR, 'val_v24.csv'), index=False)

# SNS val만 별도 저장 (모니터링용)
sns_val.to_csv(os.path.join(OUT_DIR, 'sns_val_only.csv'), index=False)

print(f'\n✅ {OUT_DIR}/ 저장 완료')
print(f'   train_v24.csv: {len(train_v24)}프레임 (real={len(train_v24[train_v24.label==0])}, fake={len(train_v24[train_v24.label==1])})')
print(f'   val_v24.csv:   {len(val_v24)}프레임 (real={len(val_v24[val_v24.label==0])}, fake={len(val_v24[val_v24.label==1])})')
print(f'   sns_val_only.csv: {len(sns_val)}프레임 (SNS 도메인 검증 전용)')
