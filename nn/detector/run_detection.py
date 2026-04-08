#!/usr/bin/env python3
"""
OpenPCDet TransFusion-L로 nuScenes 전체 추론 → bbox/detections.json

Pre-requisites
--------------
1. OpenPCDet clone + 의존성 설치:
       bash nn/detector/install_deps.sh
2. TransFusion-L 체크포인트 다운로드:
       nn/detector/OpenPCDet/checkpoints/transfusion_lidar.pth
3. nuScenes data prep (최초 1회):
       cd nn/detector/OpenPCDet
       ln -sf /data1/nuScenes data/nuscenes
       python -m pcdet.datasets.nuscenes.nuscenes_dataset \\
           --func create_nuscenes_infos \\
           --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \\
           --version v1.0-trainval

Output
------
bbox/detections.json  — nuScenes detection format, global frame
  {
    "results": {
      "<sample_token>": [
        {"translation": [x,y,z], "size": [w,l,h], "rotation": [w,x,y,z],
         "detection_name": "car", "detection_score": 0.9, "velocity": [vx,vy]}
      ]
    }
  }

Usage
-----
    python nn/detector/run_detection.py \\
        --data-root /data1/nuScenes \\
        --ckpt nn/detector/OpenPCDet/checkpoints/transfusion_lidar.pth \\
        --gpus 0 \\
        --out bbox/detections.json
"""

import os
import sys
import argparse
import subprocess
import glob
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PCDET_DIR  = os.path.join(SCRIPT_DIR, 'OpenPCDet')
CFG_FILE   = os.path.join(PCDET_DIR, 'tools/cfgs/nuscenes_models/transfusion_lidar.yaml')


def main():
    ap = argparse.ArgumentParser(description='Run TransFusion-L detection on nuScenes')
    ap.add_argument('--data-root',  default='/data1/nuScenes', help='nuScenes dataset root')
    ap.add_argument('--ckpt',       required=True, help='TransFusion-L checkpoint (.pth)')
    ap.add_argument('--gpus',       default='0', help='Comma-separated GPU ids, e.g. "0" or "0,1"')
    ap.add_argument('--batch-size', default='4', help='Inference batch size')
    ap.add_argument('--out',        default='bbox/detections.json', help='Output JSON path')
    args = ap.parse_args()

    if not os.path.isdir(PCDET_DIR):
        sys.exit(f'ERROR: OpenPCDet not found at {PCDET_DIR}\n'
                 'Run run_all.sh first to clone the repo.')
    if not os.path.isfile(CFG_FILE):
        sys.exit(f'ERROR: TransFusion-L config not found: {CFG_FILE}')
    if not os.path.isfile(args.ckpt):
        sys.exit(f'ERROR: Checkpoint not found: {args.ckpt}')

    os.makedirs('bbox', exist_ok=True)

    # Ensure data/nuscenes symlink exists inside OpenPCDet
    data_link = os.path.join(PCDET_DIR, 'data/nuscenes')
    if not os.path.exists(data_link):
        os.makedirs(os.path.join(PCDET_DIR, 'data'), exist_ok=True)
        os.symlink(args.data_root, data_link)
        print(f'Symlink created: {data_link} → {args.data_root}')

    gpu_ids = args.gpus.split(',')
    n_gpus  = len(gpu_ids)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus

    if n_gpus > 1:
        cmd = [
            'bash', os.path.join(PCDET_DIR, 'scripts/dist_test.sh'),
            str(n_gpus),
            '--cfg_file', CFG_FILE,
            '--ckpt', args.ckpt,
            '--batch_size', args.batch_size,
        ]
    else:
        cmd = [
            sys.executable, os.path.join(PCDET_DIR, 'tools/test.py'),
            '--cfg_file', CFG_FILE,
            '--ckpt', args.ckpt,
            '--batch_size', args.batch_size,
        ]

    print(f'[Detection] Running TransFusion-L on {n_gpus} GPU(s) ...')
    subprocess.run(cmd, check=True, cwd=PCDET_DIR, env=env)

    # Find the most recently written results_nusc.json under output/
    pattern = os.path.join(PCDET_DIR, 'output', '**', 'results_nusc.json')
    candidates = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not candidates:
        sys.exit('ERROR: results_nusc.json not found in OpenPCDet/output/\n'
                 'Check OpenPCDet logs for errors.')
    shutil.copy(candidates[-1], args.out)
    print(f'\nDetection results saved → {args.out}')
    print('Next step:  python nn/detector/run_tracking.py --det bbox/detections.json')


if __name__ == '__main__':
    main()
