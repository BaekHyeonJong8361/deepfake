"""
eval_v37_ood.py
---------------
DINOv2-base v37 모델 OOD 평가 (빠른 진단용)
GPU 1번, mediapipe face crop, 140 OOD 비디오
"""
import os, sys, cv2, torch, time
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import mediapipe as mp
from torchvision import transforms
from transformers import AutoModel
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
DEVICE = torch.device('cuda:0')

BASE = '/home/t26106/deepfake'
IMG_SIZE = 224
MAX_FRAMES = 5
FRAME_STEP = 30

FOLDERS = [
    (os.path.join(BASE, 'final/ood_final_eval/real'),        0),
    (os.path.join(BASE, 'test_inputs/real'),                 0),
    (os.path.join(BASE, 'final/ood_final_eval_v2/real_new'), 0),
    (os.path.join(BASE, 'final/ood_final_eval/fake'),        1),
    (os.path.join(BASE, 'test_inputs/fake'),                 1),
    (os.path.join(BASE, 'final/ood_final_eval_v2/fake_new'), 1),
]

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

class DINOv2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained('facebook/dinov2-base')
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        out = self.backbone(x).last_hidden_state[:, 0]
        return self.head(out).squeeze(1)

def load_model():
    m = DINOv2Classifier().to(DEVICE)
    ck = torch.load(os.path.join(BASE, 'saved_models/dino_best_v37_fullfinetune.pth'),
                    map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    epoch = ck.get('epoch', '?')
    val_auc = ck.get('val_auc', '?')
    print(f'  v37 loaded (ep{epoch}, val_auc={val_auc:.4f})')
    return m

def score_video(path, model, fd):
    cap = cv2.VideoCapture(path)
    frames, idx, collected = [], 0, 0
    while collected < MAX_FRAMES:
        ret, f = cap.read()
        if not ret: break
        if idx % FRAME_STEP == 0:
            frames.append(f); collected += 1
        idx += 1
    cap.release()

    scores = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if not res.detections: continue
        det = max(res.detections, key=lambda d: d.score[0])
        b = det.location_data.relative_bounding_box
        H, W = rgb.shape[:2]
        cx = (b.xmin + b.width/2) * W
        cy = (b.ymin + b.height/2) * H
        s = max(b.width*W, b.height*H)
        x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2))
        x2, y2 = int(min(W, cx+s/2)), int(min(H, cy+s/2))
        face = Image.fromarray(rgb[y1:y2, x1:x2])
        if face.width < 10 or face.height < 10: continue
        t = TFM(face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(t)
            prob = torch.sigmoid(out).item()
        scores.append(prob)

    return float(np.mean(scores)) if scores else 0.5

if __name__ == '__main__':
    t0 = time.time()
    videos = []
    for folder, label in FOLDERS:
        if not os.path.isdir(folder): continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(('.mp4','.mov','.avi','.mkv')):
                videos.append((os.path.join(folder, fn), label))

    print(f'OOD 비디오: {len(videos)}개 (real={sum(1 for _,l in videos if l==0)}, fake={sum(1 for _,l in videos if l==1)})')
    model = load_model()
    fd = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    labels, scores = [], []
    for i, (path, label) in enumerate(videos):
        s = score_video(path, model, fd)
        scores.append(s)
        labels.append(label)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{len(videos)} 완료...')

    fd.close()
    auc = roc_auc_score(labels, scores)

    real_scores = [s for s, l in zip(scores, labels) if l == 0]
    fake_scores = [s for s, l in zip(scores, labels) if l == 1]

    print(f'\n=== v37 OOD 결과 ===')
    print(f'AUC       : {auc:.4f}')
    print(f'Real mean : {np.mean(real_scores):.4f}  (낮을수록 좋음)')
    print(f'Fake mean : {np.mean(fake_scores):.4f}  (높을수록 좋음)')
    print(f'Gap       : {np.mean(fake_scores)-np.mean(real_scores):+.4f}')
    print(f'소요      : {(time.time()-t0)/60:.1f}분')
    print(f'\n[기준] v20=0.832, v27=0.699, v24+v25+v29 앙상블=0.890')
