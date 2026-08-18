"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

LDR (Ours) offline evaluation: reproduce the paper's position/radius numbers (pos = abs_x + abs_y).
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn as nn, h5py, imageio.v3 as iio
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ldr import build_ldr
from ldr.metrics import (evaluate_xy, evaluate_xy_collision, evaluate_xy_looming,
                         WORLD_SCALE, parse_state_from_image, parse_state_from_image_collision)

# Released ckpts are model-only; build the fixed LDR (Ours) architecture for them.
OURS_ARCH = dict(num_pred=29, width=256, accel_scale=0.5, n_kp=16, warp_flow_res=64, kappa_init=0.15)


def build_model(ckpt, img_size, device):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    arch = dict(OURS_ARCH)
    cargs = ck.get('args') if isinstance(ck, dict) else None
    if cargs:
        arch.update(width=cargs.get('accel_width', 256), n_kp=cargs.get('n_kp', 16),
                    accel_scale=cargs.get('accel_scale', 0.5),
                    kappa_init=cargs.get('kappa_init', cargs.get('accel_init_damp', 0.15)))
    model = build_ldr(**arch).to(device).eval()
    sd = ck['model'] if (isinstance(ck, dict) and 'model' in ck) else ck
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def gen(model, frames_t, frames_np, nc, device):
    cond_t = frames_t[:nc].to(device)
    with torch.no_grad():
        pred = model(cond_t.unsqueeze(0), nc, full=False, cond_img=cond_t[nc - 1:nc])
        pf = ((pred.squeeze(0).clamp(-1, 1) + 1) * 127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    return np.concatenate([frames_np[:nc], pf], axis=0)


def rad_single(frames, gt_pos, gt_r, color=0):
    errs = []; n = min(len(frames), len(gt_pos))
    for i in range(3, n):
        x, y = float(gt_pos[i, 0]), float(gt_pos[i, 1])
        if x - gt_r >= 0 and x + gt_r <= WORLD_SCALE and y - gt_r >= 0 and y + gt_r <= WORLD_SCALE:
            ps = parse_state_from_image(frames[i], np.nan, np.nan, color=color)
            errs.append(abs(float(ps[0, 2]) - gt_r))
    return float(np.mean(errs)) if errs else np.nan


def rad_coll(frames, gt_pos, r1, r2, post_from=10**9):
    errs = []; errs_post = []; n = min(len(frames), len(gt_pos))
    for i in range(3, n):
        ps = parse_state_from_image_collision(frames[i], np.nan, np.nan, np.nan)
        e0 = abs(float(ps[0, 2]) - r1); e1 = abs(float(ps[1, 2]) - r2)
        errs.append(e0); errs.append(e1)
        if i >= post_from:
            errs_post.append(e0); errs_post.append(e1)
    full = float(np.mean(errs)) if errs else np.nan
    post = float(np.mean(errs_post)) if errs_post else np.nan
    return full, post


def evaluate_split(model, task, eval_data, split, img_size, num_cond=3, device='cuda', eval_n=0, progress=False):
    """Evaluate on a split's ID/OOD clips; eval_n>0 caps clips per split for a fast in-training probe."""
    subset = json.load(open(split)) if isinstance(split, str) else list(split)
    if eval_n:
        subset = [s for s in subset if s[2]][:eval_n] + [s for s in subset if not s[2]][:eval_n]
    id_pos, ood_pos, id_rad, ood_rad = [], [], [], []
    id_pf, ood_pf, id_rp, ood_rp = [], [], [], []
    with h5py.File(eval_data, 'r') as f:
        for gk, li, is_id in tqdm(subset, desc=f'eval {task} @{img_size}', disable=not progress):
            try:
                raw = f['video_streams'][gk][li]
                fr = iio.imread(raw.tobytes(), index=None, format_hint='.mp4')
                ft = (torch.from_numpy(fr.astype(np.float32)).permute(0, 3, 1, 2) / 127.5 - 1.0)
                if ft.shape[-1] != img_size:
                    ft = nn.functional.interpolate(ft, size=(img_size, img_size),
                                                   mode='bilinear', align_corners=False)
                    fr = (((ft + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
                roll = gen(model, ft, fr, num_cond, device)
                gt = np.array(f['position_streams'][gk][li], dtype=np.float64)
                gt[..., 1] = WORLD_SCALE - gt[..., 1]
                init = f['init_streams'][gk][li]; n = min(len(roll), len(gt))
                if len(init) == 4:                                       # collision (two balls)
                    ret = evaluate_xy_collision(roll[:n], gt[:n], init, mode='all')
                    pos = ret['post_x_err_avg'] + ret['post_y_err_avg']
                    (id_pf if is_id else ood_pf).append(ret['abs_x_err_avg'] + ret['abs_y_err_avg'])
                    rad, radp = rad_coll(roll[:n], gt[:n], float(init[0]), float(init[1]),
                                         post_from=ret['collision_index'] + 1)
                    (id_rp if is_id else ood_rp).append(radp)
                elif len(init) == 3:                                     # looming
                    grad = np.array(f['radius_streams'][gk][li], dtype=np.float64)
                    ret = evaluate_xy_looming(roll[:n], gt[:n], grad[:n], init, mode='all')
                    pos = ret['abs_x_err_avg'] + ret['abs_y_err_avg']
                    rad = ret['abs_r_err_avg']
                else:                                                    # uniform / parabola / bouncing
                    ret = evaluate_xy(roll[:n], gt[:n], init, mode='all')
                    pos = ret['abs_x_err_avg'] + ret['abs_y_err_avg']
                    rad = rad_single(roll[:n], gt[:n], float(init[0]))
                (id_pos if is_id else ood_pos).append(pos)
                (id_rad if is_id else ood_rad).append(rad)
            except Exception as e:
                print('skip', gk, li, repr(e), file=sys.stderr)
    m = lambda x: float(np.nanmean(x)) if len(x) else float('nan')
    out = {'task': task, 'img_size': img_size, 'n_id': len(id_pos), 'n_ood': len(ood_pos),
           'id_pos': round(m(id_pos), 4), 'ood_pos': round(m(ood_pos), 4),
           'id_rad': round(m(id_rad), 4), 'ood_rad': round(m(ood_rad), 4)}
    if id_pf or ood_pf:
        out['id_pos_full'] = round(m(id_pf), 4); out['ood_pos_full'] = round(m(ood_pf), 4)
    if id_rp or ood_rp:
        out['id_rad_post'] = round(m(id_rp), 4); out['ood_rad_post'] = round(m(ood_rp), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--task', required=True,
                    choices=['uniform', 'parabola', 'collision', 'bouncing', 'looming'])
    ap.add_argument('--eval_data', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--img_size', type=int, default=128)
    ap.add_argument('--num_cond', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='logs/evaluation', help='save the result JSON here (one per ckpt+task)')
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(a.ckpt, a.img_size, device)
    out = evaluate_split(model, a.task, a.eval_data, a.split, a.img_size, a.num_cond, device, progress=True)
    out.update(ckpt=a.ckpt, eval_data=a.eval_data, split=a.split)
    print(json.dumps(out))
    os.makedirs(a.out_dir, exist_ok=True)
    tag = 'joint' if 'joint' in a.ckpt.lower() else 'single'
    out_path = os.path.join(a.out_dir, f'{a.task}_{a.img_size}_{tag}.json')
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'saved -> {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
