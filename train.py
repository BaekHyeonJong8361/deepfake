import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
import timm

from dataset import FastDeepfakeDataset, get_xception_transforms


# =========================================================
# 유틸
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        drop_last=False,
    )


def safe_roc_auc(y_true, y_prob):
    try:
        if len(set(y_true)) < 2:
            return 0.0
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.0


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            total_loss += loss.item() * images.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    auc = safe_roc_auc(all_labels, all_probs)

    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "bal_acc": bal_acc,
        "auc": auc,
    }


# =========================================================
# 메인 학습
# =========================================================
def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print(f"🔥 학습 시작 | device={device}")

    # -------------------------
    # 하이퍼파라미터
    # -------------------------
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4
    patience = 4
    num_workers = min(8, os.cpu_count() or 4)

    train_csv = "splits/dataset_train.csv"
    val_csv = "splits/dataset_val.csv"
    test_csv = "splits/dataset_test.csv"

    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "xception_best.pth")

    # -------------------------
    # 데이터셋 / 로더
    # -------------------------
    print("📁 데이터 로딩 중...")
    train_dataset = FastDeepfakeDataset(
        train_csv,
        transform=get_xception_transforms(train=True),
        label_dtype=torch.float32,
    )
    val_dataset = FastDeepfakeDataset(
        val_csv,
        transform=get_xception_transforms(train=False),
        label_dtype=torch.float32,
    )
    test_dataset = FastDeepfakeDataset(
        test_csv,
        transform=get_xception_transforms(train=False),
        label_dtype=torch.float32,
    )

    train_loader = build_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = build_loader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = build_loader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    print(f"- train samples: {len(train_dataset)}")
    print(f"- val samples:   {len(val_dataset)}")
    print(f"- test samples:  {len(test_dataset)}")

    # -------------------------
    # 모델
    # -------------------------
    print("🤖 Xception 모델 로딩 중...")
    model = timm.create_model("xception", pretrained=True, num_classes=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
    )

    use_amp = False  # cuDNN 호환 이슈로 AMP 비활성화
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_score = -1.0
    early_stop_count = 0

    # -------------------------
    # 학습 루프
    # -------------------------
    print("🚀 Epoch 시작")

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * images.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"\\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(
            f"Val   Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['acc']*100:.2f}% | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"BalAcc: {val_metrics['bal_acc']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        # 불균형 검증셋이므로 F1보다 AUC를 우선 기준으로 저장하는 것도 좋음
        current_score = val_metrics["auc"]
        scheduler.step(current_score)

        if current_score > best_score:
            print(f"⭐ 베스트 모델 갱신! ({best_score:.4f} -> {current_score:.4f})")
            best_score = current_score
            early_stop_count = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_auc": best_score,
                "val_metrics": val_metrics,
            }, best_model_path)
        else:
            early_stop_count += 1
            print(f"- early_stop_count: {early_stop_count}/{patience}")

            if early_stop_count >= patience:
                print(f"🛑 {patience}회 연속 개선 없음. 조기 종료.")
                break

    # -------------------------
    # 베스트 모델 로드 후 test 평가
    # -------------------------
    print("\\n📦 베스트 모델 로드 후 TEST 평가")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)

    print("\\n[TEST RESULT]")
    print(
        f"Loss: {test_metrics['loss']:.4f} | "
        f"Acc: {test_metrics['acc']*100:.2f}% | "
        f"F1: {test_metrics['f1']:.4f} | "
        f"BalAcc: {test_metrics['bal_acc']:.4f} | "
        f"AUC: {test_metrics['auc']:.4f}"
    )

    print(f"\\n✅ 학습 완료. 베스트 모델 저장 위치: {best_model_path}")


if __name__ == "__main__":
    train()
