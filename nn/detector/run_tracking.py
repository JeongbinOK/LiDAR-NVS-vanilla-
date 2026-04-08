#!/usr/bin/env python3
"""
MCTrack으로 detection 결과 트래킹 → bbox/tracking.json 저장.

Pre-requisites
--------------
1. MCTrack submodule 초기화:
       git submodule update --init nn/detector/MCTrack
2. MCTrack 의존성 설치:
       cd nn/detector/MCTrack && pip install -r requirements.txt

Output
------
bbox/tracking.json  — nuScenes tracking format, global frame
  {
    "results": {
      "<sample_token>": [
        {"tracking_id": "1", "translation": [x,y,z], "size": [w,l,h],
         "rotation": [w,x,y,z], "velocity": [vx,vy],
         "tracking_name": "car", "tracking_score": 0.87}
      ]
    },
    "meta": {...}
  }

tracking_id는 씬 내에서 프레임 간 persistent string ID.
데이터로더에서 이 값을 integer instance ID로 변환하여 사용.

Usage
-----
    python nn/detector/run_tracking.py \\
        --det bbox/detections.json \\
        --data-root /data1/nuScenes \\
        --out bbox/tracking.json
"""

import os
import sys
import argparse
import subprocess

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MCTrack')


def main():
    parser = argparse.ArgumentParser(description='Run MCTrack on nuScenes detections')
    parser.add_argument('--det', default='bbox/detections.json',
                        help='Path to detections.json (nuScenes detection format)')
    parser.add_argument('--data-root', default='/data1/nuScenes',
                        help='nuScenes dataset root')
    parser.add_argument('--out', default='bbox/tracking.json',
                        help='Output JSON path (nuScenes tracking format)')
    args = parser.parse_args()

    # Verify submodule is initialised
    main_script = os.path.join(REPO_DIR, 'tools', 'main.py')
    if not os.path.isfile(main_script):
        sys.exit(
            f'ERROR: MCTrack main.py not found at {main_script}\n'
            'Run:  git submodule update --init nn/detector/MCTrack'
        )

    if not os.path.isfile(args.det):
        sys.exit(
            f'ERROR: Detection file not found: {args.det}\n'
            'Run detection first:  python nn/detector/run_detection.py ...'
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'mctrack_base')

    # Step 1: Convert detection JSON → MCTrack BaseVersion format
    # MCTrack provides a conversion script in tools/
    create_data_script = os.path.join(REPO_DIR, 'tools', 'nusc_create_data.py')
    if not os.path.isfile(create_data_script):
        # Fallback: try alternative script name used in some MCTrack versions
        create_data_script = os.path.join(REPO_DIR, 'tools', 'create_data.py')

    convert_cmd = [
        sys.executable, create_data_script,
        '--det', os.path.abspath(args.det),
        '--dataroot', args.data_root,
        '--out', base_dir,
    ]
    print(f'[Step 1] Converting detections to MCTrack BaseVersion format ...')
    print(f'Running: {" ".join(convert_cmd)}')
    subprocess.run(convert_cmd, check=True, cwd=REPO_DIR)

    # Step 2: Run MCTrack tracking
    cfg_path = os.path.join(REPO_DIR, 'configs', 'nuscenes.yaml')
    if not os.path.isfile(cfg_path):
        # Try alternative config path
        cfg_path = os.path.join(REPO_DIR, 'config', 'nuscenes.yaml')

    track_cmd = [
        sys.executable, main_script,
        '--data_path', base_dir,
        '--config', cfg_path,
        '--out', os.path.abspath(args.out),
    ]
    print(f'\n[Step 2] Running MCTrack ...')
    print(f'Running: {" ".join(track_cmd)}')
    subprocess.run(track_cmd, check=True, cwd=REPO_DIR)

    print(f'\nTracking results saved → {args.out}')
    print('Next step:  Update config.py → bbox_json_path = "bbox/tracking.json"')


if __name__ == '__main__':
    main()
