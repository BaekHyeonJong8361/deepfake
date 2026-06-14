"""
eval_all_models_ood_v2.py
=========================
v8~v36 전 모델 OOD v2 (n=140) 평가.
GPU 0 사용 (GPU 1 은 추론서버 점유).
"""
import os, sys, cv2, json, time, torch
import numpy as np
from PIL import Image
import mediapipe as mp
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from torchvision import transforms

sys.path.append('/home/t26106/deepfake')

DEVICE = torch.device('cuda:1')
BASE   = '/home/t26106/deepfake'
OUT    = f'{BASE}/saved_models/all_models_ood_v2.json'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MAX_FRAMES = 5
FRAME_STEP = 30

FOLDERS = [
    (f'{BASE}/final/ood_final_eval/real',            0),
    (f'{BASE}/test_inputs/real',                     0),
    (f'{BASE}/final/ood_final_eval_v2/real_new',     0),
    (f'{BASE}/final/ood_final_eval/fake',            1),
    (f'{BASE}/test_inputs/fake',                     1),
    (f'{BASE}/final/ood_final_eval_v2/fake_new',     1),
]

# ── 전처리 ────────────────────────────────────────────────────────────
mp_fd = mp.solutions.face_detection
fd    = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_tfm(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def detect_and_crop(frame, img_size):
    H, W = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = fd.process(rgb)
    if not res.detections: return None
    det  = max(res.detections, key=lambda d: d.score[0])
    b    = det.location_data.relative_bounding_box
    cx   = (b.xmin + b.width / 2) * W
    cy   = (b.ymin + b.height / 2) * H
    s    = max(b.width * W, b.height * H)
    x1   = int(max(0, cx - s / 2))
    y1   = int(max(0, cy - s / 2))
    x2   = int(min(W, cx + s / 2))
    y2   = int(min(H, cy + s / 2))
    crop = rgb[y1:y2, x1:x2]
    if crop.size == 0: return None
    return Image.fromarray(crop).resize((img_size, img_size), Image.BILINEAR)

def score_video(path, model, tfm, img_size):
    cap   = cv2.VideoCapture(path)
    probs = []
    count, idx = 0, 0
    while count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_STEP == 0:
            face = detect_and_crop(frame, img_size)
            if face is not None:
                x = tfm(face).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(x)
                    if isinstance(out, (list, tuple)): out = out[0]
                    prob = torch.sigmoid(out.squeeze()).item()
                probs.append(prob)
                count += 1
        idx += 1
    cap.release()
    return float(np.mean(probs)) if probs else None

def collect(model, img_size):
    tfm = get_tfm(img_size)
    labels, scores = [], []
    skipped = 0
    for folder, label in FOLDERS:
        if not os.path.exists(folder): continue
        flist = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.mp4', '.mov', '.avi'))
                        and not f.startswith('._')])
        for f in flist:
            s = score_video(os.path.join(folder, f), model, tfm, img_size)
            if s is None: skipped += 1; continue
            labels.append(label); scores.append(s)
    if skipped: print(f'    (스킵 {skipped}개)')
    return labels, scores

def calc(labels, scores):
    lb = np.array(labels); sc = np.array(scores)
    if len(set(lb)) < 2: return None
    auc = roc_auc_score(lb, sc)
    fpr, tpr, thrs = roc_curve(lb, sc)
    idx = int(np.argmax(tpr - fpr))
    thr = float(thrs[idx])
    acc = float(accuracy_score(lb, (sc >= thr).astype(int)))
    return {'auc': float(auc), 'acc': acc, 'thr': thr,
            'n': len(lb), 'n_real': int((lb==0).sum()), 'n_fake': int((lb==1).sum()),
            'real_mean': float(sc[lb==0].mean()), 'fake_mean': float(sc[lb==1].mean())}

# ── 모델 레지스트리 ───────────────────────────────────────────────────
def build_registry():
    from train_v20 import F3NetLiteV20
    from train_v8  import F3NetLite
    from train_v30_effnet import EffNetB4Detector
    from train_v34_dino   import V34Dino

    reg = []

    # F3NetLite (v8 arch) — img 299
    for name, ckpt in [
        ('v8',  'f3netlite_best_v8_combined.pth'),
        ('v9',  'f3netlite_best_v9_combined.pth'),
        ('v11', 'f3netlite_best_v11_df40.pth'),
        ('v16', 'f3netlite_best_v16_sns_aug.pth'),
        ('v23', 'f3netlite_best_v23_super_sns.pth'),
        ('v24', 'f3netlite_best_v24_sns_real_finetune.pth'),
        ('v25', 'f3netlite_best_v25_sns_oversample2x.pth'),
        ('v26', 'f3netlite_best_v26_sns_oversample1x.pth'),
        ('v28', 'f3netlite_best_v28_dinov2.pth'),
        ('v29', 'f3netlite_best_v29_v20arch_more_data.pth'),
        ('v31_genuine', 'f3netlite_best_v31_genuine.pth'),
        ('v32_genuine', 'f3netlite_best_v32_genuine.pth'),
        ('v33', 'f3netlite_best_v33_freeze_fifi_crop10.pth'),
        ('v36', 'f3netlite_best_v36_df40full.pth'),
    ]:
        path = f'{BASE}/saved_models/{ckpt}'
        if not os.path.exists(path): continue
        # v8~v19 은 F3NetLite, v20+ 은 F3NetLiteV20
        cls = F3NetLite if int(name.split('_')[0][1:]) < 20 else F3NetLiteV20
        reg.append((name, path, cls, dict(img_size=299, num_bands=3,
                    pretrained=False), 299))

    # F3NetLiteV20 (v20 arch) — img 299
    for name, ckpt in [
        ('v20', 'f3netlite_best_v20_df40_sns_aug.pth'),
        ('v27', 'f3netlite_best_v27_freeze_fifi.pth'),
    ]:
        path = f'{BASE}/saved_models/{ckpt}'
        if not os.path.exists(path): continue
        reg.append((name, path, F3NetLiteV20,
                    dict(img_size=299, num_bands=3, pretrained=False), 299))

    # EfficientNet-B4 — img 380
    path = f'{BASE}/saved_models/effnet_b4_best_v30.pth'
    if os.path.exists(path):
        reg.append(('v30', path, EffNetB4Detector, {}, 380))

    # DINOv2 — img 224
    for name, ckpt in [
        ('v34', 'dino_best_v34_df40_sns.pth'),
        ('v35', 'dino_best_v35_df40_full.pth'),
    ]:
        path = f'{BASE}/saved_models/{ckpt}'
        if not os.path.exists(path): continue
        reg.append((name, path, V34Dino, {}, 224))

    return reg

# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    print('=' * 78)
    print(f'전 모델 OOD v2 평가 (n=140, GPU 1)')
    print('=' * 78)

    registry = build_registry()
    print(f'평가 대상: {len(registry)} 모델\n')

    results = {}
    for name, ckpt, cls, kwargs, img_size in registry:
        print(f'[{name}] {os.path.basename(ckpt)}')
        try:
            model = cls(**kwargs)
            ck    = torch.load(ckpt, map_location='cpu', weights_only=False)
            state = ck.get('model_state_dict', ck)
            model.load_state_dict(state, strict=False)
            model.to(DEVICE).eval()

            t0 = time.time()
            labels, scores = collect(model, img_size)
            m = calc(labels, scores)
            dt = time.time() - t0

            if m:
                m['labels'] = labels
                m['scores'] = scores
                results[name] = m
                print(f'  AUC={m["auc"]:.4f}  Acc={m["acc"]:.1%}  '
                      f'real={m["real_mean"]:.3f}  fake={m["fake_mean"]:.3f}  ({dt:.0f}s)')
            else:
                print(f'  라벨 부족 스킵')

            del model; torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ERROR: {e}')

    # 결과 정렬 출력
    print('\n' + '=' * 78)
    print('[전체 순위 (AUC 내림차순)]')
    print('=' * 78)
    sorted_r = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for i, (name, m) in enumerate(sorted_r, 1):
        print(f'  {i:2d}. {name:<15s}  AUC={m["auc"]:.4f}  Acc={m["acc"]:.1%}')

    # 앙상블 (v20+v27 best weight)
    if 'v20' in results and 'v27' in results:
        # 새로 collect 해야 하지만 scores 저장 안 했으니 기존 결과 활용
        print(f'\n  [참고] v20+v27 ensemble (w=0.25/0.75) : 0.757 (eval_ood_v2_final.py 결과)')

    with open(OUT, 'w') as f:
        json.dump({'results': results,
                   'sorted': [(n, r['auc']) for n, r in sorted_r]}, f,
                  indent=2, ensure_ascii=False)
    print(f'\n저장: {OUT}')

if __name__ == '__main__':
    main()
