#!/bin/bash
# nn/detector/run_all.sh
#
# 오프라인 전처리 파이프라인: TransFusion-L 검출 + MCTrack 트래킹
# 결과는 bbox/tracking.json 에 저장됨 (sample_token으로 인덱싱)
#
# 최초 실행 시:
#   1) pcdet conda env 생성 및 의존성 설치:
#        conda create -n pcdet python=3.8 -y && conda activate pcdet
#        bash nn/detector/install_deps.sh
#   2) TransFusion-L 체크포인트 수동 다운로드:
#        → OpenPCDet repo README → nuScenes → TransFusion-L (LiDAR-only)
#        → 저장 경로: nn/detector/OpenPCDet/checkpoints/transfusion_lidar.pth
#   3) nuScenes data prep (최초 1회, pcdet env 활성화 상태에서):
#        cd nn/detector/OpenPCDet
#        ln -sf /data1/nuScenes data/nuscenes
#        python -m pcdet.datasets.nuscenes.nuscenes_dataset \
#            --func create_nuscenes_infos \
#            --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
#            --version v1.0-trainval
#
# 이후 실행 (체크포인트 + 의존성 + data prep 완료 후):
#   conda activate pcdet
#   bash nn/detector/run_all.sh [--gpus <id>]
#
# --gpus 옵션:
#   --gpus 0      → GPU 0번 사용 (기본값)
#   --gpus 4      → GPU 4번 사용
#   --gpus 0,1    → GPU 0, 1번 멀티GPU 사용

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PCDET_DIR="$SCRIPT_DIR/OpenPCDet"
MCTRACK_DIR="$SCRIPT_DIR/MCTrack"

# ── config.py에서 data_root 읽기 (CLI로 override 가능) ──────────────────────
DATAROOT=$(python -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
from config import QGSConfig
print(QGSConfig().data_root)
" 2>/dev/null) || DATAROOT="/data1/nuScenes"

CKPT="$PCDET_DIR/checkpoints/transfusion_lidar.pth"
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
if [ ! -d "$PCDET_DIR/.git" ]; then
    echo "[setup] Cloning OpenPCDet ..."
    git clone https://github.com/open-mmlab/OpenPCDet.git "$PCDET_DIR"
else
    echo "[setup] OpenPCDet already present."
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
    echo "ERROR: TransFusion-L 체크포인트가 없습니다."
    echo "  저장 위치: $CKPT"
    echo "  다운로드:  OpenPCDet repo README → nuScenes → TransFusion-L (LiDAR-only)"
    exit 1
fi

echo ""
echo "========================================"
echo " TransFusion-L + MCTrack 전처리 파이프라인"
echo "========================================"
echo "  data-root : $DATAROOT"
echo "  ckpt      : $CKPT"
echo "  gpus      : $GPUS"
echo ""

cd "$REPO_ROOT"

# ── Step 1: 검출 ──────────────────────────────────────────────────────────
echo "[1/2] TransFusion-L 검출 ..."
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
