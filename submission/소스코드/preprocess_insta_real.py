"""
preprocess_insta_real.py
insta/ 디렉터리의 SNS real 영상 → 얼굴 크롭 → preprocessed_faces/sns_real/
CSV 엔트리도 생성하여 기존 splits와 병합 가능하도록 함
"""
import os, sys, cv2, numpy as np
from PIL import Image
import mediapipe as mp

INPUT_DIR  = '/home/t26106/deepfake/insta'
OUTPUT_DIR = '/home/t26106/deepfake/preprocessed_faces/sns_real'
CSV_OUT    = '/home/t26106/deepfake/splits/sns_real_frames.csv'
TARGET_FPS = 5      # 초당 5프레임 추출
MAX_FRAMES = 60     # 영상당 최대 프레임 수
FACE_SIZE  = 299
MIN_BLUR   = 30.0   # 너무 흐린 프레임 제외

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)

mp_fd = mp.solutions.face_detection
fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def detect_and_crop(frame):
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)
    if not res.detections:
        return None
    det = max(res.detections, key=lambda d: d.score[0])
    b = det.location_data.relative_bounding_box
    cx, cy = (b.xmin + b.width/2)*W, (b.ymin + b.height/2)*H
    s = max(b.width*W, b.height*H) * 1.3
    x1, y1 = int(max(0, cx-s/2)), int(max(0, cy-s/2))
    x2, y2 = int(min(W, cx+s/2)), int(min(H, cy+s/2))
    if x2 <= x1 or y2 <= y1:
        return None
    face = Image.fromarray(rgb[y1:y2, x1:x2]).resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    return face

def is_blurry(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR

def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / TARGET_FPS)))

    video_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_dir, exist_ok=True)

    saved, idx, frame_idx = 0, 0, 0
    entries = []

    while saved < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            if not is_blurry(frame):
                face = detect_and_crop(frame)
                if face is not None:
                    fname = f'frame_{frame_idx:04d}.jpg'
                    fpath = os.path.join(video_dir, fname)
                    face.save(fpath, quality=95)
                    entries.append({
                        'split': 'train',
                        'image_path': fpath,
                        'label': 0,
                        'group': 'original',
                        'subtype': 'sns_real',
                        'video_name': video_name,
                        'holdout_key': abs(hash(video_name)) % 100,
                        'frame_dir': video_dir,
                        'dataset': 'sns'
                    })
                    saved += 1
                    frame_idx += 1
        idx += 1

    cap.release()
    return saved, entries

def main():
    videos = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(('.mp4', '.mov', '.avi'))
    ])

    if not videos:
        print(f'[ERROR] {INPUT_DIR} 에 영상 없음')
        return

    print(f'총 {len(videos)}개 영상 전처리 시작')
    print(f'출력: {OUTPUT_DIR}')
    print('='*50)

    all_entries = []
    total_frames = 0

    for i, vname in enumerate(videos):
        vpath = os.path.join(INPUT_DIR, vname)
        vkey = os.path.splitext(vname)[0]  # 확장자 제거
        saved, entries = process_video(vpath, vkey)
        all_entries.extend(entries)
        total_frames += saved
        print(f'[{i+1:2d}/{len(videos)}] {vname[:40]:<40} → {saved}프레임')

    # CSV 저장
    import csv
    with open(CSV_OUT, 'w', newline='') as f:
        fieldnames = ['split','image_path','label','group','subtype','video_name','holdout_key','frame_dir','dataset']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    print('='*50)
    print(f'완료: {len(videos)}개 영상 → {total_frames}프레임 저장')
    print(f'CSV: {CSV_OUT}')

if __name__ == '__main__':
    main()
