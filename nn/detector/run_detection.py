#!/usr/bin/env python3
"""
OpenPCDet TransFusion-L로 nuScenes 추론 → bbox/detections_{split}.json

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
nuScenes detection format (global frame):
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
        --split val \\
        --data-root /data1/nuScenes \\
        --ckpt nn/detector/OpenPCDet/checkpoints/transfusion_lidar.pth \\
        --gpus 0 \\
        --out bbox/detections_val.json
"""

import os
import sys
import argparse
import subprocess
import glob
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PCDET_DIR  = os.path.join(SCRIPT_DIR, 'OpenPCDet')
TOOLS_DIR  = os.path.join(PCDET_DIR, 'tools')
# _BASE_CONFIG_ in transfusion_lidar.yaml uses paths relative to tools/,
# so test.py must be run with cwd=tools/ — use relative cfg_file accordingly
CFG_FILE   = 'cfgs/nuscenes_models/transfusion_lidar.yaml'


def main():
    ap = argparse.ArgumentParser(description='Run TransFusion-L detection on nuScenes')
    ap.add_argument('--split',      default='val', choices=['val', 'train'],
                    help='Dataset split to run inference on')
    ap.add_argument('--data-root',  default='/data1/nuScenes', help='nuScenes dataset root')
    ap.add_argument('--ckpt',       required=True, help='TransFusion-L checkpoint (.pth)')
    ap.add_argument('--gpus',       default='0', help='Comma-separated GPU ids, e.g. "0" or "0,1"')
    ap.add_argument('--batch-size', default='4', help='Inference batch size')
    ap.add_argument('--out',        default='bbox/detections_val.json', help='Output JSON path')
    args = ap.parse_args()

    cfg_abs = os.path.join(TOOLS_DIR, CFG_FILE)
    if not os.path.isdir(PCDET_DIR):
        sys.exit(f'ERROR: OpenPCDet not found at {PCDET_DIR}\n'
                 'Run run_all.sh first to clone the repo.')
    if not os.path.isfile(cfg_abs):
        sys.exit(f'ERROR: TransFusion-L config not found: {cfg_abs}')
    if not os.path.isfile(args.ckpt):
        sys.exit(f'ERROR: Checkpoint not found: {args.ckpt}')

    os.makedirs('bbox', exist_ok=True)

    # OpenPCDet expects: data/nuscenes → {data_root}
    # and {data_root}/v1.0-trainval/samples → {data_root}/samples  (same for sweeps)
    # because NuScenesDataset appends VERSION to root_path, so lidar lookup becomes:
    #   data/nuscenes/v1.0-trainval/samples/LIDAR_TOP/...
    data_link = os.path.join(PCDET_DIR, 'data/nuscenes')
    if not os.path.exists(data_link):
        os.makedirs(os.path.join(PCDET_DIR, 'data'), exist_ok=True)
        os.symlink(args.data_root, data_link)
        print(f'Symlink created: {data_link} → {args.data_root}')

    version = 'v1.0-trainval'
    for subdir in ('samples', 'sweeps'):
        inner = os.path.join(args.data_root, version, subdir)
        if not os.path.exists(inner):
            target = os.path.join(args.data_root, subdir)
            os.symlink(target, inner)
            print(f'Symlink created: {inner} → {target}')

    gpu_ids = args.gpus.split(',')
    n_gpus  = len(gpu_ids)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus

    # extra_tag separates output dirs so val/train results don't overwrite each other
    extra_tag = f'split_{args.split}'

    # All invocations use cwd=TOOLS_DIR so that _BASE_CONFIG_ relative paths
    # (e.g. "cfgs/dataset_configs/nuscenes_dataset.yaml") resolve correctly
    if n_gpus > 1:
        cmd = [
            sys.executable, '-m', 'torch.distributed.launch',
            f'--nproc_per_node={n_gpus}',
            'test.py',   # relative to TOOLS_DIR
            '--launcher', 'pytorch',
            '--cfg_file', CFG_FILE,
            '--ckpt', args.ckpt,
            '--batch_size', args.batch_size,
            '--extra_tag', extra_tag,
        ]
    else:
        cmd = [
            sys.executable, 'test.py',   # relative to TOOLS_DIR
            '--cfg_file', CFG_FILE,
            '--ckpt', args.ckpt,
            '--batch_size', args.batch_size,
            '--extra_tag', extra_tag,
        ]

    # train split: override dataset config so test.py runs on train pkl
    # literal_eval requires inner quotes: "['file.pkl']" → ['file.pkl'] (list)
    # "[file.pkl]" without inner quotes → literal_eval fails → treated as string → 0 samples
    if args.split == 'train':
        cmd += [
            '--set',
            'DATA_CONFIG.DATA_SPLIT.test', 'train',
            'DATA_CONFIG.INFO_PATH.test', "['nuscenes_infos_10sweeps_train.pkl']",
        ]

    print(f'[Detection] Running TransFusion-L (split={args.split}) on {n_gpus} GPU(s) ...')
    # check=False: NuScenesEval may fail for train split (eval_set is hardcoded to 'val'),
    # but results_nusc.json is saved before metrics computation — we verify it exists below.
    ret = subprocess.run(cmd, cwd=TOOLS_DIR, env=env)
    if ret.returncode != 0:
        print(f'WARNING: test.py exited with code {ret.returncode} '
              f'(likely NuScenesEval on {args.split} split — non-critical if results_nusc.json exists)')

    # OpenPCDet writes output relative to cfg.ROOT_DIR = PCDET_DIR
    # path: PCDET_DIR/output/{model}/{extra_tag}/eval/.../results_nusc.json
    pattern = os.path.join(PCDET_DIR, 'output', '**', extra_tag, '**', 'results_nusc.json')
    candidates = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not candidates:
        # fallback: most recent results_nusc.json anywhere under output/
        pattern = os.path.join(PCDET_DIR, 'output', '**', 'results_nusc.json')
        candidates = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    if not candidates:
        sys.exit('ERROR: results_nusc.json not found in OpenPCDet/output/\n'
                 'Check OpenPCDet logs for errors.')
    shutil.copy(candidates[-1], args.out)
    print(f'\nDetection results saved → {args.out}  (split={args.split})')
    print(f'Next step:  python nn/detector/run_tracking.py --split {args.split} --det {args.out}')


if __name__ == '__main__':
    main()
