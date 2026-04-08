#!/bin/bash
# nn/detector/run_all.sh
#
# 오프라인 전처리 파이프라인: LargeKernel3D 검출 + MCTrack 트래킹
# 결과는 bbox/tracking.json 에 저장됨 (sample_token으로 인덱싱)
#
# 최초 실행 시:
#   1) LargeKernel3D와 MCTrack이 없으면 자동으로 clone
#   2) LK3D 체크포인트를 수동으로 다운로드 필요
#      → LargeKernel3D repo README의 Google Drive 링크
#      → 저장 경로: nn/detector/LargeKernel3D/checkpoints/lk3d_nuscenes.pth
#   3) 의존성 설치 (별도 conda env 권장, lnvs와 충돌 가능)
#      → bash nn/detector/install_deps.sh
#
# 이후 실행 (체크포인트 + 의존성 설치 완료 후):
#   bash nn/detector/run_all.sh [--gpus <id>]
#
# --gpus 옵션:
#   --gpus 0      → GPU 0번 사용 (기본값)
#   --gpus 4      → GPU 4번 사용
#   --gpus 0,1    → GPU 0, 1번 멀티GPU 사용

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── config.py에서 data_root 읽기 (CLI로 override 가능) ──────────────────────
DATAROOT=$(python -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from config import QGSConfig
print(QGSConfig().data_root)
" 2>/dev/null) || DATAROOT="/data1/nuScenes"

CKPT="$SCRIPT_DIR/LargeKernel3D/checkpoints/lk3d_nuscenes.pth"
GPUS=0

# CLI override
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpus)      GPUS="$2";      shift ;;
        --data-root) DATAROOT="$2";  shift ;;
        --ckpt)      CKPT="$2";      shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── 외부 repo 자동 clone ───────────────────────────────────────────────────
LK3D_DIR="$SCRIPT_DIR/LargeKernel3D"
MCTRACK_DIR="$SCRIPT_DIR/MCTrack"

if [ ! -d "$LK3D_DIR/.git" ]; then
    echo "[setup] Cloning LargeKernel3D ..."
    git clone https://github.com/dvlab-research/LargeKernel3D.git "$LK3D_DIR"
else
    echo "[setup] LargeKernel3D already present."
fi

if [ ! -d "$MCTRACK_DIR/.git" ]; then
    echo "[setup] Cloning MCTrack ..."
    git clone https://github.com/megvii-research/MCTrack.git "$MCTRACK_DIR"
else
    echo "[setup] MCTrack already present."
fi

# ── 체크포인트 확인 ────────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo ""
    echo "ERROR: LargeKernel3D 체크포인트가 없습니다."
    echo "  저장 위치: $CKPT"
    echo "  다운로드:  LargeKernel3D repo README → Google Drive 링크"
    echo "             (nuScenes val: 69.1 NDS / 63.3 mAP 모델)"
    exit 1
fi

echo ""
echo "========================================"
echo " LargeKernel3D + MCTrack 전처리 파이프라인"
echo "========================================"
echo "  data-root : $DATAROOT"
echo "  ckpt      : $CKPT"
echo "  gpus      : $GPUS"
echo ""

cd "$REPO_ROOT"

# ── Step 1: 검출 ──────────────────────────────────────────────────────────
echo "[1/2] LargeKernel3D 검출 ..."
python nn/detector/run_detection.py \
    --data-root "$DATAROOT" \
    --ckpt "$CKPT" \
    --gpus "$GPUS" \
    --out bbox/detections.json

# ── Step 2: 트래킹 ────────────────────────────────────────────────────────
echo ""
echo "[2/2] MCTrack 트래킹 ..."
python nn/detector/run_tracking.py \
    --det bbox/detections.json \
    --data-root "$DATAROOT" \
    --out bbox/tracking.json

echo ""
echo "========================================"
echo " 완료: bbox/tracking.json"
echo " 사용: config.py → bbox_json_path = 'bbox/tracking.json'"
echo "========================================"
