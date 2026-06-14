"""
hyperparam_sweep_v2.py
----------------------
아직 시도 안 한 4가지 축 실험:
  1. min_detection_confidence: 0.3 / 0.4 / 0.5 (현재)
  2. v26 앙상블 포함 조합
  3. 가중치 세밀 그리드 (0.05 단위, v24+v25+v29 근방)
  4. FRAME_STEP: 20 / 25 / 30 (현재)

기준: OOD v2 140개 (final/ood_final_eval + test_inputs + ood_final_eval_v2)
결과: saved_models/hyperparam_sweep_v2.json
"""
import os, sys, json, itertools, cv2, torch, time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import mediapipe as mp
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0')   # VISIBLE 1번 = 내부 0번

sys.path.append('/home/t26106/deepfake')
from train_v20 import F3NetLiteV20

BASE = '/home/t26106/deepfake'
FACE_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CROP_SCALE    = 1.0

FOLDERS = [
    (os.path.join(BASE, 'final/ood_final_eval/real'),           0),
    (os.path.join(BASE, 'test_inputs/real'),                    0),
    (os.path.join(BASE, 'final/ood_final_eval_v2/real_new'),    0),
    (os.path.join(BASE, 'final/ood_final_eval/fake'),           1),
    (os.path.join(BASE, 'test_inputs/fake'),                    1),
    (os.path.join(BASE, 'final/ood_final_eval_v2/fake_new'),    1),
]

TFM = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ------------------------------------------------------------------
# 모델 로딩
# ------------------------------------------------------------------
CKPT = {
    'v24': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
    'v25': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
    'v26': 'saved_models/f3netlite_best_v26_sns_oversample1x.pth',
    'v29': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
}

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def load_models():
    models = {}
    for name, rel_path in CKPT.items():
        path = os.path.join(BASE, rel_path)
        m = F3NetLiteV20(img_size=FACE_SIZE, num_bands=3, pretrained=False).to(DEVICE)
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        m.eval()
        models[name] = m
        print(f'  ✅ {name} loaded')
    return models

# ------------------------------------------------------------------
# 비디오 → raw 확률 배열 (캐싱)
# ------------------------------------------------------------------
def get_raw_probs(path, models, fd, frame_step, max_frames=5):
    cap = cv2.VideoCapture(path)
    frames, idx, collected = [], 0, 0
    while collected < max_frames:
        ret, f = cap.read()
        if not ret: break
        if idx % frame_step == 0:
            frames.append(f); collected += 1
        idx += 1
    cap.release()

    probs = {k: [] for k in models}
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if not res.detections:
            continue
        det = max(res.detections, key=lambda d: d.score[0])
        b = det.location_data.relative_bounding_box
        H, W = rgb.shape[:2]
        cx, cy = (b.xmin + b.width/2)*W, (b.ymin + b.height/2)*H
        s = max(b.width*W, b.height*H) * CROP_SCALE
        x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2))
        x2, y2 = int(min(W, cx+s/2)), int(min(H, cy+s/2))
        face = Image.fromarray(rgb[y1:y2, x1:x2])
        if face.width < 10 or face.height < 10:
            continue
        t = TFM(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            for k, m in models.items():
                out = m(t)
                if isinstance(out, (list, tuple)): out = out[0]
                probs[k].append(float(sigmoid_np(out.squeeze().cpu().numpy())))
    return probs

def ensemble_score(probs_dict, weights):
    # weights: dict {model_name: w}
    avail = {k: np.array(v) for k, v in probs_dict.items() if len(v) > 0 and k in weights}
    if not avail:
        return 0.5
    total_w = sum(weights[k] for k in avail)
    norm_w  = {k: weights[k] / total_w for k in avail}
    n = max(len(v) for v in avail.values())
    ens = np.zeros(n)
    for k, arr in avail.items():
        padded = np.pad(arr, (0, n - len(arr)), mode='edge') if len(arr) < n else arr[:n]
        ens += norm_w[k] * padded
    return float(np.mean(ens))

# ------------------------------------------------------------------
# 데이터셋 수집
# ------------------------------------------------------------------
def collect_videos():
    videos = []
    for folder, label in FOLDERS:
        if not os.path.isdir(folder): continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                videos.append((os.path.join(folder, fn), label))
    return videos

# ------------------------------------------------------------------
# 실험 1: min_detection_confidence
# ------------------------------------------------------------------
def exp1_detection_conf(videos, models):
    print('\n' + '='*60)
    print('EXP 1: min_detection_confidence sweep')
    print('='*60)
    results = {}
    base_weights = {'v24': 0.2, 'v25': 0.5, 'v29': 0.3}

    for conf in [0.3, 0.4, 0.5]:
        fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=conf)
        scores, labels = [], []
        for path, label in videos:
            probs = get_raw_probs(path, models, fd, frame_step=30)
            scores.append(ensemble_score(probs, base_weights))
            labels.append(label)
        fd.close()
        auc = roc_auc_score(labels, scores)
        print(f'  conf={conf:.1f}  AUC={auc:.4f}')
        results[f'conf_{conf}'] = {'conf': conf, 'auc': auc}
    return results

# ------------------------------------------------------------------
# 실험 2: v26 포함 조합
# ------------------------------------------------------------------
def exp2_v26(videos, models):
    print('\n' + '='*60)
    print('EXP 2: v26 inclusion')
    print('='*60)
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    # 비디오당 raw probs 한 번만 계산
    all_probs = []
    labels = []
    for path, label in videos:
        probs = get_raw_probs(path, models, fd, frame_step=30)
        all_probs.append(probs)
        labels.append(label)
    fd.close()

    combos = [
        # 기존 baseline
        {'v24': 0.2, 'v25': 0.5, 'v29': 0.3},
        # v26 추가 4-way
        {'v24': 0.2, 'v25': 0.4, 'v26': 0.1, 'v29': 0.3},
        {'v24': 0.2, 'v25': 0.3, 'v26': 0.2, 'v29': 0.3},
        {'v24': 0.15, 'v25': 0.5, 'v26': 0.1, 'v29': 0.25},
        # v26 3-way (v29 제거)
        {'v24': 0.3, 'v25': 0.5, 'v26': 0.2},
        {'v24': 0.2, 'v25': 0.5, 'v26': 0.3},
        # v26 3-way (v24 제거)
        {'v25': 0.5, 'v26': 0.2, 'v29': 0.3},
        {'v25': 0.6, 'v26': 0.2, 'v29': 0.2},
    ]

    results = {}
    for w in combos:
        scores = [ensemble_score(p, w) for p in all_probs]
        auc = roc_auc_score(labels, scores)
        label_str = '+'.join(f'{k}({v})' for k, v in w.items())
        print(f'  {label_str:<45}  AUC={auc:.4f}')
        results[label_str] = {'weights': w, 'auc': auc}
    return results

# ------------------------------------------------------------------
# 실험 3: 가중치 세밀 그리드 (v24+v25+v29, 0.05 단위)
# ------------------------------------------------------------------
def exp3_fine_grid(videos, models):
    print('\n' + '='*60)
    print('EXP 3: fine weight grid (v24+v25+v29, step=0.05)')
    print('='*60)
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)

    all_probs, labels = [], []
    for path, label in videos:
        probs = get_raw_probs(path, models, fd, frame_step=30)
        all_probs.append(probs)
        labels.append(label)
    fd.close()

    best_auc, best_w = 0, None
    results = {}
    steps = [round(x * 0.05, 2) for x in range(1, 18)]   # 0.05 ~ 0.85
    for w24 in steps:
        for w25 in steps:
            w29 = round(1.0 - w24 - w25, 2)
            if w29 < 0.05 or w29 > 0.85: continue
            w = {'v24': w24, 'v25': w25, 'v29': w29}
            scores = [ensemble_score(p, w) for p in all_probs]
            auc = roc_auc_score(labels, scores)
            k = f'{w24}/{w25}/{w29}'
            results[k] = auc
            if auc > best_auc:
                best_auc = auc; best_w = w
    print(f'  Grid size: {len(results)} combos')
    top5 = sorted(results.items(), key=lambda x: -x[1])[:5]
    for k, v in top5:
        print(f'  v24/v25/v29={k}  AUC={v:.4f}')
    print(f'  ★ BEST: {best_w}  AUC={best_auc:.4f}')
    return {'best': {'weights': best_w, 'auc': best_auc}, 'all': results}

# ------------------------------------------------------------------
# 실험 4: FRAME_STEP
# ------------------------------------------------------------------
def exp4_frame_step(videos, models):
    print('\n' + '='*60)
    print('EXP 4: FRAME_STEP sweep')
    print('='*60)
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    base_weights = {'v24': 0.2, 'v25': 0.5, 'v29': 0.3}
    results = {}
    for step in [15, 20, 25, 30]:
        scores, labels = [], []
        for path, label in videos:
            probs = get_raw_probs(path, models, fd, frame_step=step)
            scores.append(ensemble_score(probs, base_weights))
            labels.append(label)
        auc = roc_auc_score(labels, scores)
        print(f'  FRAME_STEP={step}  AUC={auc:.4f}')
        results[f'step_{step}'] = {'step': step, 'auc': auc}
    fd.close()
    return results

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == '__main__':
    t0 = time.time()
    print('[hyperparam_sweep_v2] 시작')

    videos = collect_videos()
    print(f'비디오 {len(videos)}개 (real={sum(1 for _,l in videos if l==0)}, fake={sum(1 for _,l in videos if l==1)})')

    models = load_models()

    out = {}
    out['exp1_detection_conf'] = exp1_detection_conf(videos, models)
    out['exp2_v26']            = exp2_v26(videos, models)
    out['exp3_fine_grid']      = exp3_fine_grid(videos, models)
    out['exp4_frame_step']     = exp4_frame_step(videos, models)

    elapsed = time.time() - t0
    print(f'\n총 소요: {elapsed/60:.1f}분')

    out_path = os.path.join(BASE, 'saved_models/hyperparam_sweep_v2.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'결과 저장: {out_path}')

    # 최종 요약
    print('\n=== 최종 요약 ===')
    e1 = out['exp1_detection_conf']
    best_conf = max(e1.values(), key=lambda x: x['auc'])
    print(f'EXP1 최고 conf: {best_conf["conf"]}  AUC={best_conf["auc"]:.4f}')

    e2 = out['exp2_v26']
    best_v26 = max(e2.values(), key=lambda x: x['auc'])
    print(f'EXP2 최고 v26조합: AUC={best_v26["auc"]:.4f}  {best_v26["weights"]}')

    e3 = out['exp3_fine_grid']
    print(f'EXP3 최고 가중치: {e3["best"]["weights"]}  AUC={e3["best"]["auc"]:.4f}')

    e4 = out['exp4_frame_step']
    best_step = max(e4.values(), key=lambda x: x['auc'])
    print(f'EXP4 최고 step: {best_step["step"]}  AUC={best_step["auc"]:.4f}')
