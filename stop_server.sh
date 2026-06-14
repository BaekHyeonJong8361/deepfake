#!/bin/bash
pkill -f "inference_server_v10" 2>/dev/null && echo "✅ 추론 서버 종료"
pkill -f "ssh.*60006" 2>/dev/null && echo "✅ SSH 터널 종료"
