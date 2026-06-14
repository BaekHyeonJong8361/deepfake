"""
train_v37.py — DINOv2-base Full Fine-tune
==========================================
v34/v35 와의 결정적 차이: backbone을 frozen하지 않고 전체 학습.
- Backbone (DINOv2-base, 87M): lr=5e-6 (매우 낮게 — catastrophic forgetting 방지)
- Head (Linear 768→1):         lr=1e-3 (높게 — 빠른 수렴)
- Warmup 1 epoch 후 cosine decay

변경 변수 (v35 대비):
  - DINOv2 frozen=False (이게 핵심)
  - 아키텍처: DINOv2-base + Linear head (주파수 branch 없음 — 단순 검증용)
  - 데이터: splits_v35 (243K, SNS 22%) 재사용
  - 이미지 크기: 224×224 (DINOv2 기본, CLIP 표준)

실행:
  source ~/.venv/bin/activate
  nohup python -u train_v37.py > train_v37.log 2>&1 &
"""
import os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoModel
from sklearn.metrics import roc_auc_score
from PIL import Image

# ============================================================
# 환경 설정
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 반드시 1번 GPU

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False  # cuDNN 충돌 방지 (학과서버)

set_seed(42)
device = torch.device('cuda:0')  # VISIBLE 1번 = 내부 0번
print(f'[v37] device={device}')

# ============================================================
# 하이퍼파라미터
# ============================================================
BACKBONE_LR = 5e-6   # DINOv2 backbone: 매우 낮게
HEAD_LR     = 1e-3   # Classification head
BATCH_SIZE  = 32
EPOCHS      = 15
IMG_SIZE    = 224    # DINOv2 표준
WARMUP_EP   = 1
SNS_OVERSAMPLE = 3   # SNS 샘플 3배 오버샘플링

TRAIN_CSV = '/home/t26106/deepfake/splits_v35/train_v35.csv'
VAL_CSV   = '/home/t26106/deepfake/splits_v35/val_v35.csv'
SAVE_PATH = '/home/t26106/deepfake/saved_models/dino_best_v37_fullfinetune.pth'
LOG_PATH  = '/home/t26106/deepfake/train_v37.log'

# DINOv2 표준 정규화 (mean=std=0.5)
TFM_TRAIN = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
TFM_VAL = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ============================================================
# 데이터셋
# ============================================================
class FaceDataset(Dataset):
    def __init__(self, csv_path, tfm, sns_oversample=1):
        df = pd.read_csv(csv_path)
        if sns_oversample > 1:
            sns_rows = df[df['dataset'] == 'sns']
            df = pd.concat([df] + [sns_rows] * (sns_oversample - 1), ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.paths  = df['image_path'].tolist()
        self.labels = df['label'].tolist()
        self.tfm    = tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.tfm(img), torch.tensor(self.labels[idx], dtype=torch.float32)
        except:
            img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            return self.tfm(img), torch.tensor(0.0)

# ============================================================
# 모델
# ============================================================
class DINOv2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        hidden = self.backbone.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]  # CLS token
        return self.head(out).squeeze(1)

# ============================================================
# 학습
# ============================================================
def train():
    train_ds = FaceDataset(TRAIN_CSV, TFM_TRAIN, sns_oversample=SNS_OVERSAMPLE)
    val_ds   = FaceDataset(VAL_CSV,   TFM_VAL,   sns_oversample=1)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = DINOv2Classifier().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'총 파라미터: {total_params:.1f}M  |  학습 가능: {trainable:.1f}M')

    # Differential LR
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': BACKBONE_LR, 'weight_decay': 1e-2},
        {'params': model.head.parameters(),     'lr': HEAD_LR,     'weight_decay': 1e-4},
    ])

    # Cosine annealing (warmup 포함)
    def lr_lambda(ep):
        if ep < WARMUP_EP:
            return ep / max(WARMUP_EP, 1)
        progress = (ep - WARMUP_EP) / (EPOCHS - WARMUP_EP)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # SNS 비율 고려 pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5).to(device))

    best_auc, best_ep = 0.0, 0
    log = open(LOG_PATH, 'w')

    def logprint(s):
        print(s); log.write(s + '\n'); log.flush()

    logprint(f'[v37] DINOv2-base Full Fine-tune 시작')
    logprint(f'train={len(train_ds)}, val={len(val_ds)}, epochs={EPOCHS}')
    logprint(f'backbone_lr={BACKBONE_LR}, head_lr={HEAD_LR}, batch={BATCH_SIZE}')

    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        losses = []
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        all_scores, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                probs = torch.sigmoid(out).cpu().numpy()
                all_scores.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_auc = roc_auc_score(all_labels, all_scores)
        elapsed = time.time() - t0
        bb_lr = optimizer.param_groups[0]['lr']
        hd_lr = optimizer.param_groups[1]['lr']

        logprint(f'EP{ep:02d} | train_loss={np.mean(losses):.4f} | '
                 f'val_loss={val_loss/len(val_dl):.4f} | val_auc={val_auc:.4f} | '
                 f'bb_lr={bb_lr:.2e} | hd_lr={hd_lr:.2e} | {elapsed:.0f}s')

        if val_auc > best_auc:
            best_auc = val_auc; best_ep = ep
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': ep, 'val_auc': val_auc,
                        'config': {'backbone': 'dinov2-base', 'img_size': IMG_SIZE,
                                   'backbone_lr': BACKBONE_LR, 'head_lr': HEAD_LR}},
                       SAVE_PATH)
            logprint(f'  ★ Best 저장 (ep{ep}, AUC={val_auc:.4f})')

    logprint(f'\n완료 | Best: ep{best_ep}, val_auc={best_auc:.4f}')
    logprint(f'저장: {SAVE_PATH}')
    log.close()

if __name__ == '__main__':
    train()
