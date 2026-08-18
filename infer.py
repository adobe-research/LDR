"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

LDR inference: roll out future frames from conditioning frames and save an MP4.
"""
import os, sys, argparse, glob
import numpy as np, torch, torch.nn as nn, h5py, imageio.v3 as iio, imageio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval import build_model, gen


def _resize(ft, fr, img_size):
    if ft.shape[-1] != img_size:
        ft = nn.functional.interpolate(ft, size=(img_size, img_size), mode='bilinear', align_corners=False)
        fr = (((ft + 1) * 127.5).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
    return ft, fr


def load_clip(eval_data, group, index, img_size):
    with h5py.File(eval_data, 'r') as f:
        raw = f['video_streams'][group][index]
    fr = iio.imread(raw.tobytes(), index=None, format_hint='.mp4')
    ft = (torch.from_numpy(fr.astype(np.float32)).permute(0, 3, 1, 2) / 127.5 - 1.0)
    return _resize(ft, fr, img_size)


def load_frames_dir(frames_dir, num_cond, img_size):
    paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
    assert len(paths) >= num_cond, f'{frames_dir}: need >= {num_cond} conditioning images, found {len(paths)}'
    fr = np.stack([np.asarray(iio.imread(p))[..., :3] for p in paths[:num_cond]], 0)
    ft = (torch.from_numpy(fr.astype(np.float32)).permute(0, 3, 1, 2) / 127.5 - 1.0)
    return _resize(ft, fr, img_size)


def error_map(gt, pred, gain=4.0):
    """Per-pixel GT-vs-pred error as a black->red->yellow->white colormap."""
    e = np.abs(gt.astype(np.float32) - pred.astype(np.float32)).mean(-1)
    e = np.clip(e * gain / 255.0, 0.0, 1.0)
    r = np.clip(e * 3.0, 0, 1); g = np.clip(e * 3.0 - 1, 0, 1); b = np.clip(e * 3.0 - 2, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--frames_dir', default=None, help='dir of conditioning frame images (00.png, 01.png, ...)')
    ap.add_argument('--eval_data', default=None, help='PhyWorld .hdf5 clip (alternative to --frames_dir)')
    ap.add_argument('--group', default='00000')
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--img_size', type=int, default=256, help="inference resolution (the checkpoint's training resolution)")
    ap.add_argument('--out_size', type=int, default=None, help='upscale saved frames to this size (frames_dir mode; default = img_size)')
    ap.add_argument('--num_cond', type=int, default=3)
    ap.add_argument('--out', default='pred.mp4')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--side_by_side', action='store_true', help='hdf5 mode only: write GT | error map | prediction')
    a = ap.parse_args()
    assert a.frames_dir or a.eval_data, 'provide --frames_dir (conditioning images) or --eval_data (.hdf5 clip)'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(a.ckpt, a.img_size, device)
    if a.frames_dir:
        ft, fr = load_frames_dir(a.frames_dir, a.num_cond, a.img_size)
    else:
        ft, fr = load_clip(a.eval_data, a.group, a.index, a.img_size)
    pred = gen(model, ft, fr, a.num_cond, device)
    if a.out_size and not a.side_by_side and a.out_size != pred.shape[1]:
        t = nn.functional.interpolate(torch.from_numpy(pred).permute(0, 3, 1, 2).float(),
                                      size=(a.out_size, a.out_size), mode='bilinear', align_corners=False)
        pred = t.permute(0, 2, 3, 1).round().clamp(0, 255).byte().numpy()
    if a.side_by_side and a.eval_data:
        n = min(len(fr), len(pred))
        gt, pr = fr[:n], pred[:n]
        pred = np.concatenate([gt, error_map(gt, pr), pr], axis=2)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(a.out, list(pred), fps=a.fps, codec='libx264')
    print(f'wrote {a.out}  ({len(pred)} frames, {pred.shape[2]}x{pred.shape[1]})')


if __name__ == '__main__':
    main()
