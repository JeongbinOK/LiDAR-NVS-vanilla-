#!/usr/bin/env python3
"""
MCTrack으로 detection 결과 트래킹 → bbox/tracking.json

Pre-requisites
--------------
1. MCTrack clone + 의존성 설치:
       bash nn/detector/install_deps.sh
2. bbox/detections.json 생성:
       python nn/detector/run_detection.py ...

MCTrack 파이프라인
-----------------
1. OpenPCDet results_nusc.json → bbox/det_input/transfusion/val.json 으로 복사
   (convert_nuscenes.py가 내부에서 {dets_path}/{detector}/{split}.json 경로 조합)
2. preprocess/convert_nuscenes.py::nuscenes_main() 직접 호출 → BaseVersion JSON 생성
3. config/nuscenes.yaml 수정 (--config 옵션 없음, 파일 직접 수정)
4. python main.py --dataset nuscenes -e -p 8
   (main.py가 SAVE_PATH를 {parent}/nuscenes/YYYYMMDD_HHMMSS/ 로 override)
5. 타임스탬프 디렉토리에서 results.json 찾아서 args.out으로 복사

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

Usage
-----
    python nn/detector/run_tracking.py \\
        --det bbox/detections.json \\
        --data-root /data1/nuScenes \\
        --split val \\
        --out bbox/tracking.json
"""

import os
import sys
import argparse
import shutil
import yaml
import subprocess
import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MCTRACK_DIR = os.path.join(SCRIPT_DIR, 'MCTrack')
DETECTOR    = 'transfusion'


def main():
    ap = argparse.ArgumentParser(description='Run MCTrack on nuScenes detections')
    ap.add_argument('--det',       required=True, help='detections.json (OpenPCDet results_nusc.json)')
    ap.add_argument('--data-root', default='/data1/nuScenes', help='nuScenes dataset root')
    ap.add_argument('--split',     default='val', help='val | test')
    ap.add_argument('--out',       default='bbox/tracking.json', help='Output JSON path')
    args = ap.parse_args()

    if not os.path.isdir(MCTRACK_DIR):
        sys.exit(f'ERROR: MCTrack not found at {MCTRACK_DIR}\n'
                 'Run run_all.sh first to clone the repo.')
    if not os.path.isfile(args.det):
        sys.exit(f'ERROR: Detection file not found: {args.det}\n'
                 'Run detection first: python nn/detector/run_detection.py ...')

    os.makedirs('bbox', exist_ok=True)

    # Step 1: OpenPCDet 결과를 MCTrack이 기대하는 경로 구조로 복사
    # convert_nuscenes.py 내부: path = os.path.join(dets_path, detector, split + ".json")
    det_subdir = os.path.abspath(f'bbox/det_input/{DETECTOR}')
    os.makedirs(det_subdir, exist_ok=True)
    dest = os.path.join(det_subdir, f'{args.split}.json')
    shutil.copy(os.path.abspath(args.det), dest)
    det_root = os.path.abspath('bbox/det_input')
    print(f'[Step 1] Copied detections → {dest}')

    # Step 2: detection JSON → MCTrack BaseVersion 변환
    # nuscenes_main(raw_data_path, dets_path, detector, save_path, split) — 5개 위치 인자
    sys.path.insert(0, MCTRACK_DIR)
    from preprocess.convert_nuscenes import nuscenes_main

    base_dir = os.path.abspath('bbox/mctrack_base')
    print(f'[Step 2] Converting to MCTrack BaseVersion → {base_dir} ...')
    nuscenes_main(
        raw_data_path=args.data_root,
        dets_path=det_root,
        detector=DETECTOR,
        save_path=base_dir,
        split=args.split,
    )
    # 변환 결과: base_dir/transfusion/val.json

    # Step 3: MCTrack config/nuscenes.yaml 직접 수정 (--config 옵션 없음)
    mctrack_cfg_path = os.path.join(MCTRACK_DIR, 'config', 'nuscenes.yaml')
    with open(mctrack_cfg_path) as f:
        track_cfg = yaml.safe_load(f)
    track_cfg['DATASET_ROOT']    = args.data_root
    track_cfg['DETECTIONS_ROOT'] = base_dir   # BaseVersion JSON 경로
    track_cfg['DETECTOR']        = DETECTOR
    track_cfg['SPLIT']           = args.split
    # main.py가 SAVE_PATH를 {parent}/nuscenes/YYYYMMDD_HHMMSS/ 로 override함
    mctrack_out = os.path.abspath('bbox/mctrack_out')
    track_cfg['SAVE_PATH']       = mctrack_out
    with open(mctrack_cfg_path, 'w') as f:
        yaml.dump(track_cfg, f)
    print(f'[Step 3] Updated MCTrack config: {mctrack_cfg_path}')

    # Step 4: MCTrack 트래킹 실행
    print(f'[Step 4] Running MCTrack ...')
    subprocess.run(
        [sys.executable, 'main.py', '--dataset', 'nuscenes', '-e', '-p', '8'],
        check=True, cwd=MCTRACK_DIR
    )

    # Step 5: 결과 JSON 찾기
    # main.py: save_path = os.path.join(os.path.dirname(cfg["SAVE_PATH"]), cfg["DATASET"], timestamp)
    save_parent = os.path.dirname(mctrack_out)
    pattern = os.path.join(save_parent, 'nuscenes', '*', 'results.json')
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        sys.exit(f'ERROR: MCTrack results.json not found at {pattern}\n'
                 'Check MCTrack logs for errors.')
    shutil.copy(candidates[-1], args.out)
    print(f'\nTracking results saved → {args.out}')
    print('Next step:  Update config.py → bbox_json_path = "bbox/tracking.json"')


if __name__ == '__main__':
    main()
