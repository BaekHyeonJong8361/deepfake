"""
eval_tta.py
-----------
Test-Time Augmentation (TTA) 평가
현재 최고 앙상블 (v24+v25+v29, 0.2/0.5/0.3) 기준으로
같은 프레임을 여러 변환으로 돌려 평균 → AUC 개선 측정

TTA 변환:
  1. 원본
  2. 좌우반전
  3. 밝기 +20
  4. 밝기 -20
  5. 약간 zoom-in (center crop 90%)

기준: OOD 140개, GPU 1번
"""
import os, sys, cv2, torch, time
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.metrics import roc_auc_score
import mediapipe as mp
from torchvision import transforms

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0')

sys.path.append('/home/t26106/deepfake')
from train_v20 import F3NetLiteV20

BASE = '/home/t26106/deepfake'
FACE_SIZE = 299
CROP_SCALE = 1.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FOLDERS = [
    (os.path.join(BASE, 'final/ood_final_eval/real'),           0),
    (os.path.join(BASE, 'test_inputs/real'),                    0),
    (os.path.join(BASE, 'final/ood_final_eval_v2/real_new'),    0),
    (os.path.join(BASE, 'final/ood_final_eval/fake'),           1),
    (os.path.join(BASE, 'test_inputs/fake'),                    1),
    (os.path.join(BASE, 'final/ood_final_eval_v2/fake_new'),    1),
]

WEIGHTS = {'v24': 0.2, 'v25': 0.5, 'v29': 0.3}
CKPT = {
    'v24': 'saved_models/f3netlite_best_v24_sns_real_finetune.pth',
    'v25': 'saved_models/f3netlite_best_v25_sns_oversample2x.pth',
    'v29': 'saved_models/f3netlite_best_v29_v20arch_more_data.pth',
}

TFM = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def tta_variants(face_pil):
    """원본 포함 5가지 TTA 변환"""
    variants = [face_pil]
    variants.append(face_pil.transpose(Image.FLIP_LEFT_RIGHT))
    variants.append(ImageEnhance.Brightness(face_pil).enhance(1.3))
    variants.append(ImageEnhance.Brightness(face_pil).enhance(0.7))
    # center crop 90% → resize back
    w, h = face_pil.size
    margin = int(w * 0.05)
    cropped = face_pil.crop((margin, margin, w-margin, h-margin))
    variants.append(cropped.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR))
    return variants

def load_models():
    models = {}
    for name, rel in CKPT.items():
        m = F3NetLiteV20(img_size=FACE_SIZE, num_bands=3, pretrained=False).to(DEVICE)
        ck = torch.load(os.path.join(BASE, rel), map_location=DEVICE, weights_only=False)
        m.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        m.eval()
        models[name] = m
        print(f'  ✅ {name} loaded')
    return models

def score_video(path, models, fd, use_tta=True):
    cap = cv2.VideoCapture(path)
    frames, idx, collected = [], 0, 0
    while collected < 5:
        ret, f = cap.read()
        if not ret: break
        if idx % 30 == 0:
            frames.append(f); collected += 1
        idx += 1
    cap.release()

    all_ens = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if not res.detections: continue
        det = max(res.detections, key=lambda d: d.score[0])
        b = det.location_data.relative_bounding_box
        H, W = rgb.shape[:2]
        cx, cy = (b.xmin + b.width/2)*W, (b.ymin + b.height/2)*H
        s = max(b.width*W, b.height*H) * CROP_SCALE
        x1,y1 = int(max(0,cx-s/2)), int(max(0,cy-s/2))
        x2,y2 = int(min(W,cx+s/2)), int(min(H,cy+s/2))
        face = Image.fromarray(rgb[y1:y2, x1:x2])
        if face.width < 10 or face.height < 10: continue

        variants = tta_variants(face) if use_tta else [face]
        frame_scores = []
        for v in variants:
            t = TFM(v).unsqueeze(0).to(DEVICE)
            probs = {}
            with torch.no_grad():
                for k, m in models.items():
                    out = m(t)
                    if isinstance(out, (list,tuple)): out = out[0]
                    probs[k] = float(sigmoid_np(out.squeeze().cpu().numpy()))
            ens = sum(WEIGHTS[k] * probs[k] for k in WEIGHTS) / sum(WEIGHTS.values())
            frame_scores.append(ens)
        all_ens.append(float(np.mean(frame_scores)))

    return float(np.mean(all_ens)) if all_ens else 0.5

def collect_videos():
    videos = []
    for folder, label in FOLDERS:
        if not os.path.isdir(folder): continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.mp4','.mov','.avi','.mkv')):
                videos.append((os.path.join(folder, fn), label))
    return videos

if __name__ == '__main__':
    t0 = time.time()
    videos = collect_videos()
    print(f'비디오 {len(videos)}개')
    models = load_models()
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # baseline (TTA 없음)
    print('\n[Baseline - TTA 없음]')
    labels, scores_base = [], []
    for path, label in videos:
        labels.append(label)
        scores_base.append(score_video(path, models, fd, use_tta=False))
    auc_base = roc_auc_score(labels, scores_base)
    print(f'  AUC = {auc_base:.4f}')

    # TTA 적용
    print('\n[TTA 5종 적용]')
    scores_tta = []
    for i, (path, label) in enumerate(videos):
        scores_tta.append(score_video(path, models, fd, use_tta=True))
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{len(videos)} 완료...')
    auc_tta = roc_auc_score(labels, scores_tta)
    print(f'  AUC = {auc_tta:.4f}')

    fd.close()
    print(f'\n=== 결과 ===')
    print(f'Baseline : {auc_base:.4f}')
    print(f'TTA      : {auc_tta:.4f}  ({auc_tta-auc_base:+.4f})')
    print(f'소요: {(time.time()-t0)/60:.1f}분')
