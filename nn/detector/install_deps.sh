#!/bin/bash
# nn/detector/install_deps.sh
#
# LargeKernel3D + MCTrack 의존성 설치
# 시스템 CUDA 버전을 자동 감지하여 맞는 PyTorch wheel 선택.
#
# 사용법 (별도 env 권장):
#   conda create -n lk3d python=3.8 -y && conda activate lk3d
#   bash nn/detector/install_deps.sh
#
# 이미 PyTorch가 설치된 env(lnvs 등)에서 실행하면 PyTorch 설치는 건너뜀.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LK3D_DIR="$SCRIPT_DIR/LargeKernel3D"
MCTRACK_DIR="$SCRIPT_DIR/MCTrack"

# ── CUDA 버전 자동 감지 ────────────────────────────────────────────────────
detect_cuda_tag() {
    # nvcc가 있으면 그걸 기준으로, 없으면 nvidia-smi로 fallback
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: //' | awk '{print $1}')
    else
        echo "WARNING: CUDA를 감지할 수 없습니다. cu118로 fallback."
        echo "cu118"
        return
    fi

    MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    MINOR=$(echo "$CUDA_VER" | cut -d. -f2)

    if   [ "$MAJOR" -ge 12 ] && [ "$MINOR" -ge 4 ]; then echo "cu124"
    elif [ "$MAJOR" -ge 12 ] && [ "$MINOR" -ge 1 ]; then echo "cu121"
    elif [ "$MAJOR" -ge 12 ];                        then echo "cu121"
    elif [ "$MAJOR" -eq 11 ] && [ "$MINOR" -ge 8 ]; then echo "cu118"
    else
        echo "WARNING: CUDA $CUDA_VER는 오래된 버전입니다. cu118로 fallback."
        echo "cu118"
    fi
}

CU_TAG=$(detect_cuda_tag)
echo "감지된 CUDA tag: $CU_TAG"
echo "  (RTX 4090: 보통 cu121/cu124, RTX 3090: 보통 cu118)"

# ── PyTorch 설치 (이미 있으면 건너뜀) ─────────────────────────────────────
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo ""
    echo "PyTorch $TORCH_VER 이미 설치됨 → 건너뜀 (기존 env 사용 중)"
else
    echo ""
    echo "[1/3] PyTorch 설치 (tag: $CU_TAG) ..."
    pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/$CU_TAG"
fi

# ── mmdet3d 설치 ───────────────────────────────────────────────────────────
echo ""
echo "[2/3] mmdet3d 설치 ..."
pip install -U openmim
mim install mmengine
mim install "mmcv==2.0.0"
mim install "mmdet==3.0.0"
mim install "mmdet3d==1.2.0"

# ── LargeKernel3D 패키지 설치 ─────────────────────────────────────────────
if [ -d "$LK3D_DIR" ]; then
    echo ""
    echo "[3/3] LargeKernel3D 설치 ..."
    cd "$LK3D_DIR"
    [ -f "requirements.txt" ] && pip install -r requirements.txt
    python setup.py develop 2>/dev/null || echo "(setup.py 없음, 건너뜀)"
    cd -
else
    echo "WARNING: $LK3D_DIR 없음. run_all.sh 먼저 실행해서 clone 하세요."
fi

# ── MCTrack 의존성 ─────────────────────────────────────────────────────────
if [ -d "$MCTRACK_DIR" ]; then
    echo ""
    echo "[MCTrack] 의존성 설치 ..."
    cd "$MCTRACK_DIR"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        pip install scipy filterpy  # MCTrack 기본 의존성
    fi
    cd -
else
    echo "WARNING: $MCTRACK_DIR 없음. run_all.sh 먼저 실행해서 clone 하세요."
fi

echo ""
echo "========================================"
echo " 설치 완료 (CUDA: $CU_TAG)"
echo " 다음: 체크포인트 다운로드 후 run_all.sh 실행"
echo "   저장 위치: nn/detector/LargeKernel3D/checkpoints/lk3d_nuscenes.pth"
echo "========================================"
