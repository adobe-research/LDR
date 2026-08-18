"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

LDR multi-node training from scratch (encoder + warp decoder + residual f_theta, jointly).
"""
import os
import sys
import time
import logging
import argparse
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ldr.model import build_ldr, PerceptualPyramidLoss
from ldr.dataset import PhyWorldDataset


def setup_distributed():
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def get_logger(rank):
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        level=logging.INFO if rank == 0 else logging.WARNING)
    logger = logging.getLogger()
    _info = logger.info
    logger.info = lambda msg, *a, **kw: _info(f'[rank{rank}] ' + str(msg), *a, **kw)
    return logger


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--resume', default=None)
    p.add_argument('--eval_data', default=None, help='eval .hdf5; if set, eval ID/OOD at every saved checkpoint')
    p.add_argument('--split', default=None, help='splits/*.json listing the ID/OOD clips for --eval_data')
    p.add_argument('--task', default=None, help='task name (for the eval log line)')
    p.add_argument('--eval_n', type=int, default=64, help='ID and OOD clips to score per checkpoint (0 = full split)')
    p.add_argument('--n_kp', type=int, default=16, help='number of structured-latent keypoints')
    p.add_argument('--accel_width', type=int, default=256, help='width of the residual predictor f_theta (MLP)')
    p.add_argument('--accel_scale', type=float, default=0.5, help='per-step bound: residual = accel_scale*tanh(MLP)')
    p.add_argument('--kappa_init', type=float, default=0.15, help='initial uncertainty-gate scale for the 2nd-order init')
    p.add_argument('--num_cond', type=int, default=3)
    p.add_argument('--img_size', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=8, help='per-GPU batch size')
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--total_steps', type=int, default=10000)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lambda_ae', type=float, default=1.0, help='weight of the RGB reconstruction loss')
    p.add_argument('--lambda_consist', type=float, default=0.5, help='weight of the latent rollout loss')
    p.add_argument('--curriculum', action=argparse.BooleanOptionalAction, default=True,
                   help='grow the rollout horizon short->full during training (default on; --no-curriculum to disable)')
    p.add_argument('--curr_full_at', type=int, default=8000, help='step by which the horizon reaches the full length')
    args = p.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    np.random.seed(args.seed + rank)
    import random as _random
    _random.seed(args.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger = get_logger(rank)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = PhyWorldDataset(args.data, num_cond=args.num_cond, img_size=args.img_size, augment=True)
    num_pred = dataset[0]['target_frames'].shape[0]
    logger.info(f'Dataset: {len(dataset)} samples, num_cond={args.num_cond}, num_pred={num_pred}')
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0))

    model = build_ldr(n_kp=args.n_kp, num_pred=num_pred, width=args.accel_width,
                      accel_scale=args.accel_scale, kappa_init=args.kappa_init).to(device)
    perceptual = PerceptualPyramidLoss().to(device)
    if rank == 0:
        logger.info(f'LDR params: {sum(x.numel() for x in model.parameters())/1e6:.2f}M  perceptual={perceptual.note}')
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if world_size > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        raw_model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start_step = ck['step']
        logger.info(f'Resumed at step {start_step}')

    step = start_step
    t0 = time.time()
    running = 0.0
    accum = 0
    opt.zero_grad()
    nc = args.num_cond
    mse = nn.functional.mse_loss
    while step < args.total_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            if step >= args.total_steps:
                break
            cond = batch['cond_frames'].to(device)
            tgt = batch['target_frames'].to(device)
            frames = torch.cat([cond, tgt], dim=1)
            if args.curriculum and step < args.curr_full_at:
                stage = step // max(1, args.curr_full_at // 4)
                h = min(num_pred, 4 * (2 ** stage))
            else:
                h = num_pred
            cond_img = frames[:, nc - 1]
            dec_roll, dec_ae, roll_z, z = model(frames, nc, full=True, horizon=h, cond_img=cond_img)
            _, _, _, H_, W_ = dec_roll.shape
            l_roll = perceptual(dec_roll.reshape(-1, 3, H_, W_), frames[:, nc:nc + h].reshape(-1, 3, H_, W_))
            l_ae = perceptual(dec_ae.reshape(-1, 3, H_, W_), frames.reshape(-1, 3, H_, W_))
            l_con = mse(roll_z, z[:, nc:nc + h].detach())
            loss = (l_roll + args.lambda_ae * l_ae + args.lambda_consist * l_con) / args.grad_accum
            loss.backward()
            running += loss.item()  # per-optimizer-step mean loss (independent of grad_accum)
            accum += 1
            if accum == args.grad_accum:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                step += 1
                if rank == 0 and (step % args.log_every == 0 or step == 1):
                    n = 1 if step == 1 else args.log_every
                    logger.info(f'step={step:6d}  loss={running/n:.5f}  {n/(time.time()-t0):.1f}it/s')
                    running = 0.0
                    t0 = time.time()
                if step % args.save_every == 0 or step == 1:
                    if world_size > 1:
                        dist.barrier()
                    if rank == 0:
                        path = os.path.join(args.output_dir, f'ckpt_{step:06d}.pt')
                        torch.save({'step': step, 'model': raw_model.state_dict(),
                                    'optimizer': opt.state_dict(), 'args': vars(args)}, path)
                        logger.info(f'Saved {path}')
                        if args.eval_data and args.split:
                            try:
                                from eval import evaluate_split
                                raw_model.eval()
                                r = evaluate_split(raw_model, args.task, args.eval_data, args.split,
                                                   args.img_size, args.num_cond, device, args.eval_n)
                                logger.info('[eval step=%d] %s' % (step, '  '.join(
                                    f'{k}={v}' for k, v in r.items() if k not in ('task', 'img_size'))))
                            except Exception as e:
                                logger.warning(f'  eval skip: {e}')
                            finally:
                                raw_model.train()
                    if world_size > 1:
                        dist.barrier()

    if rank == 0:
        torch.save({'step': step, 'model': raw_model.state_dict(), 'args': vars(args)},
                   os.path.join(args.output_dir, 'ckpt_final.pt'))
        logger.info('Training done.')
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
