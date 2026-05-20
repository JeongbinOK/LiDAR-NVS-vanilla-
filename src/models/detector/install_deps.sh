#!/bin/bash
# nn/detector/install_deps.sh
#
# OpenPCDet(TransFusion-L) + MCTrack 의존성 설치
# pcdet conda env 에서 실행:
#   conda create -n pcdet python=3.8 -y && conda activate pcdet
#   bash nn/detector/install_deps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PCDET_DIR="$SCRIPT_DIR/OpenPCDet"
MCTRACK_DIR="$SCRIPT_DIR/MCTrack"

# ── PyTorch (cu118 — CUDA 11.8 toolkit, RTX 3090/4090 모두 호환) ─────────────
if python -c "import torch" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "PyTorch $TORCH_VER 이미 설치됨 → 건너뜀"
else
    echo "[1/3] PyTorch 설치 (cu118) ..."
    pip install torch==2.0.1 torchvision==0.15.2 \
        --index-url https://download.pytorch.org/whl/cu118
fi

# ── spconv (OpenPCDet sparse conv 핵심 의존성) ───────────────────────────────
echo ""
echo "[2/3] spconv + 공통 의존성 설치 ..."
pip install spconv-cu118
pip install nuscenes-devkit==1.0.5
pip install pyquaternion SharedArray tensorboardX easydict \
    pyyaml scikit-image tqdm

# ── OpenPCDet 설치 ────────────────────────────────────────────────────────────
if [ -d "$PCDET_DIR" ]; then
    echo ""
    echo "[3/3] OpenPCDet 설치 ..."
    cd "$PCDET_DIR"
    [ -f "requirements.txt" ] && pip install -r requirements.txt
    python setup.py develop
    cd -
else
    echo "WARNING: $PCDET_DIR 없음. run_all.sh 먼저 실행해서 clone 하세요."
fi

# ── MCTrack 의존성 ─────────────────────────────────────────────────────────────
# MCTrack requirements.txt는 numpy==1.22.0을 pin하지만 SharedArray와 충돌함.
# numpy==1.24.4로 고정 후 requirements에서 numpy만 제외하고 설치.
if [ -d "$MCTRACK_DIR" ]; then
    echo ""
    echo "[MCTrack] 의존성 설치 ..."
    pip install "numpy==1.24.4"  # SharedArray 호환 버전 (MCTrack도 정상 동작)
    cd "$MCTRACK_DIR"
    grep -v "^numpy" requirements.txt | pip install -r /dev/stdin
    cd -
else
    echo "WARNING: $MCTRACK_DIR 없음. run_all.sh 먼저 실행해서 clone 하세요."
fi

echo ""
echo "========================================"
echo " 설치 완료"
echo " 다음: TransFusion-L 체크포인트 다운로드"
echo "   저장 위치: nn/detector/OpenPCDet/checkpoints/transfusion_lidar.pth"
echo " 그 후: bash nn/detector/run_all.sh"
echo "========================================"
