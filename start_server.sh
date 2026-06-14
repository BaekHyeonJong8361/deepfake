#!/bin/bash
# Solomon AI 추론 서버 + SSH 터널 시작 스크립트
# 사용법: bash ~/start_server.sh

EC2_HOST="ubuntu@43.200.145.225"
SSH_KEY="$HOME/.ssh/id_ed25519"
PORT=60006
LOG="$HOME/deepfake/server_v10.log"

echo "[1/3] 추론 서버 상태 확인..."
if pgrep -f "inference_server_v10" > /dev/null; then
    echo "  ✅ 추론 서버 이미 실행 중"
else
    echo "  🚀 추론 서버 시작..."
    cd ~/deepfake
    source ~/.venv/bin/activate
    nohup python -u inference_server_v10.py > "$LOG" 2>&1 &
    sleep 8
    if pgrep -f "inference_server_v10" > /dev/null; then
        echo "  ✅ 추론 서버 시작 완료 (PID: $(pgrep -f inference_server_v10))"
    else
        echo "  ❌ 추론 서버 시작 실패 — 로그 확인: tail -20 $LOG"
        exit 1
    fi
fi

echo "[2/3] SSH 터널 상태 확인..."
if pgrep -f "ssh.*$PORT.*$EC2_HOST" > /dev/null; then
    echo "  ✅ SSH 터널 이미 연결 중"
else
    echo "  🔗 SSH 터널 연결..."
    ssh -i "$SSH_KEY" \
        -R "0.0.0.0:$PORT:localhost:$PORT" \
        -N \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        "$EC2_HOST" &
    sleep 3
    if pgrep -f "ssh.*$PORT.*$EC2_HOST" > /dev/null; then
        echo "  ✅ SSH 터널 연결 완료"
    else
        echo "  ❌ SSH 터널 연결 실패"
        exit 1
    fi
fi

echo "[3/3] 연결 테스트..."
RESULT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:$PORT/health 2>/dev/null)
if [ "$RESULT" = "200" ]; then
    echo "  ✅ 추론 서버 응답 정상"
else
    echo "  ⚠️  응답 없음 (모델 로딩 중일 수 있음, 10초 후 다시 확인)"
fi

echo ""
echo "✅ 완료. 로그 확인: tail -f $LOG"
