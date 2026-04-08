#!/usr/bin/env python3
"""
LargeKernel3D로 nuScenes 전체 추론 → bbox/detections.json 저장.

Pre-requisites
--------------
1. LargeKernel3D submodule 초기화:
       git submodule update --init nn/detector/LargeKernel3D
2. mmdet3d + LargeKernel3D 의존성 설치 (별도 env 권장):
       cd nn/detector/LargeKernel3D && pip install -r requirements.txt
       pip install -e .
3. 체크포인트 다운로드 (LargeKernel3D repo README → Google Drive 링크):
       nn/detector/LargeKernel3D/checkpoints/lk3d_nuscenes.pth

Output
------
bbox/detections.json  — nuScenes detection format, global frame
  {
    "results": {
      "<sample_token>": [
        {"translation": [x,y,z], "size": [w,l,h], "rotation": [w,x,y,z],
         "detection_name": "car", "detection_score": 0.9, "velocity": [vx,vy]}
      ]
    },
    "meta": {...}
  }

Usage
-----
    python nn/detector/run_detection.py \\
        --data-root /data1/nuScenes \\
        --ckpt nn/detector/LargeKernel3D/checkpoints/lk3d_nuscenes.pth \\
        --out bbox/detections.json
"""

import os
import sys
import argparse
import subprocess

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LargeKernel3D')
# Config path inside the LargeKernel3D repo (adjust if the repo layout differs)
CFG_PATH = os.path.join(REPO_DIR, 'configs', 'nuscenes', 'lar_detection_nuscenes.py')


def main():
    parser = argparse.ArgumentParser(description='Run LargeKernel3D detection on nuScenes')
    parser.add_argument('--data-root', default='/data1/nuScenes',
                        help='nuScenes dataset root')
    parser.add_argument('--ckpt', required=True,
                        help='Path to LargeKernel3D nuScenes checkpoint (.pth)')
    parser.add_argument('--out', default='bbox/detections.json',
                        help='Output JSON path (nuScenes detection format)')
    parser.add_argument('--gpus', default='0',
                        help='Comma-separated GPU ids, e.g. "0,1"')
    args = parser.parse_args()

    # Verify submodule is initialised
    if not os.path.isfile(CFG_PATH):
        sys.exit(
            f'ERROR: LargeKernel3D config not found at {CFG_PATH}\n'
            'Run:  git submodule update --init nn/detector/LargeKernel3D'
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # jsonfile_prefix: mmdet3d writes "<prefix>.bbox.json"; we strip the ".json" suffix
    json_prefix = args.out
    if json_prefix.endswith('.json'):
        json_prefix = json_prefix[:-5]

    gpu_list = args.gpus.split(',')
    if len(gpu_list) > 1:
        # Multi-GPU inference via mmdet3d's dist_test.sh
        script = os.path.join(REPO_DIR, 'tools', 'dist_test.sh')
        cmd = [
            'bash', script,
            CFG_PATH, args.ckpt, str(len(gpu_list)),
            '--format-only',
            '--eval-options', f'jsonfile_prefix={json_prefix}',
        ]
    else:
        # Single-GPU
        test_script = os.path.join(REPO_DIR, 'tools', 'test.py')
        cmd = [
            sys.executable, test_script,
            CFG_PATH, args.ckpt,
            '--format-only',
            '--eval-options', f'jsonfile_prefix={json_prefix}',
        ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    # Let LargeKernel3D's config pick up the data root via environment variable.
    # The config typically reads: data_root = os.environ.get('NUSCENES_DATA_ROOT', '/default/path')
    env['NUSCENES_DATA_ROOT'] = args.data_root

    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True, env=env, cwd=REPO_DIR)

    # mmdet3d appends '_NuSc_results' or similar; find the actual output file
    expected = args.out
    bbox_file = json_prefix + '.bbox.json'
    nusc_file = json_prefix + '_NuSc_results' + '.json'

    for candidate in [expected, bbox_file, nusc_file]:
        if os.path.isfile(candidate):
            if candidate != expected:
                os.rename(candidate, expected)
            break

    print(f'\nDetection results saved → {args.out}')
    print('Next step:  python nn/detector/run_tracking.py --det bbox/detections.json')


if __name__ == '__main__':
    main()
