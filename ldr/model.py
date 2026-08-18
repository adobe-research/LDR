"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

LDR model: structured-latent encoder, kinematic-integration rollout, warp-render decoder, perceptual loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _KeypointEnc(nn.Module):
    """Frame (N,3,H,W) -> K heatmaps at H/8 -> marginal soft-argmax -> structured latent (N,K,3)=(mu_x,mu_y,sigma)."""
    def __init__(self, n_kp, gn=8):
        super().__init__()
        self.n_kp = n_kp
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3), nn.GroupNorm(gn, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(gn, 64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(gn, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GroupNorm(gn, 96), nn.SiLU(),
            nn.Conv2d(96, 96, 3, 1, 1), nn.GroupNorm(gn, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, 2, 1), nn.GroupNorm(gn, 128), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(gn, 128), nn.SiLU(),
            nn.Conv2d(128, n_kp, 1))

    def forward(self, x):
        hm = self.net(x)
        H, W = hm.shape[-2:]
        gx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, W)
        gy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H)
        px = torch.softmax(hm.mean(2), dim=-1)
        py = torch.softmax(hm.mean(3), dim=-1)
        cx = (px * gx).sum(-1); cy = (py * gy).sum(-1)             # centroid mu
        vx = (px * (gx - cx.unsqueeze(-1)) ** 2).sum(-1)
        vy = (py * (gy - cy.unsqueeze(-1)) ** 2).sum(-1)
        s = torch.sqrt(0.5 * (vx + vy) + 1e-6)                     # extent sigma
        return torch.stack([cx, cy, s], dim=-1)


def _make_coord_grid(h, w, device, dtype=torch.float32):
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    xx = x.view(1, w).expand(h, w)
    yy = y.view(h, 1).expand(h, w)
    return torch.stack([xx, yy], dim=-1)


def _region_gaussian(centers, stds, res, device):
    grid = _make_coord_grid(res, res, device).view(1, 1, res, res, 2)
    c = centers.view(*centers.shape[:-1], 1, 1, 2)
    var = (stds ** 2).view(*stds.shape, 1, 1).clamp(min=1e-6)
    d2 = ((grid - c) ** 2).sum(-1)
    return torch.exp(-0.5 * d2 / var)


class _GNSameBlock(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1, gn=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(gn, cout)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class _GNDownBlock(nn.Module):
    def __init__(self, cin, cout, gn=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(gn, cout)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class _GNUpBlock(nn.Module):
    def __init__(self, cin, cout, gn=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(gn, cout)

    def forward(self, x):
        return F.relu(self.norm(self.conv(F.interpolate(x, scale_factor=2))))


class _GNResBlock(nn.Module):
    def __init__(self, c, gn=8):
        super().__init__()
        self.n1 = nn.GroupNorm(gn, c); self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(gn, c); self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        out = self.c1(F.relu(self.n1(x)))
        out = self.c2(F.relu(self.n2(out)))
        return out + x


class _HGEncoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks, max_features, gn=8):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            cin = in_features if i == 0 else min(max_features, block_expansion * (2 ** i))
            cout = min(max_features, block_expansion * (2 ** (i + 1)))
            blocks.append(_GNDownBlock(cin, cout, gn=gn))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outs = [x]
        for b in self.blocks:
            outs.append(b(outs[-1]))
        return outs


class _HGDecoder(nn.Module):
    def __init__(self, block_expansion, in_features, num_blocks, max_features, gn=8):
        super().__init__()
        ups = []
        for i in range(num_blocks)[::-1]:
            cin = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            cout = min(max_features, block_expansion * (2 ** i))
            ups.append(_GNUpBlock(cin, cout, gn=gn))
        self.ups = nn.ModuleList(ups)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up in self.ups:
            out = up(out)
            out = torch.cat([out, x.pop()], dim=1)
        return out


class _Hourglass(nn.Module):
    """Small GroupNorm U-Net used by the dense-motion mask predictor."""
    def __init__(self, block_expansion, in_features, num_blocks, max_features, gn=8):
        super().__init__()
        self.enc = _HGEncoder(block_expansion, in_features, num_blocks, max_features, gn=gn)
        self.dec = _HGDecoder(block_expansion, in_features, num_blocks, max_features, gn=gn)
        self.out_filters = self.dec.out_filters

    def forward(self, x):
        return self.dec(self.enc(x))


class _MeasuredWarpRenderer(nn.Module):
    """Warp the fixed cond frame to the predicted pose (FOMM/MRAA similarity flow + Gao splat occlusion)."""

    def __init__(self, num_kp, block_expansion=32, max_features=256, num_down_blocks=2,
                 num_bottleneck_blocks=3, flow_res=64, mask_block_expansion=32,
                 mask_num_blocks=4, mask_max_features=256,
                 occ_lo=1e-3, occ_hi=2.0, scale_min=0.04, bg_resp=0.1, gn=8):
        super().__init__()
        self.num_kp = num_kp; self.flow_res = flow_res
        self.occ_lo = occ_lo; self.occ_hi = occ_hi
        self.scale_min = scale_min; self.bg_resp = bg_resp
        in_feat = (num_kp + 1) * (3 + 1)
        self.hourglass = _Hourglass(mask_block_expansion, in_feat, mask_num_blocks, mask_max_features, gn=gn)
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, 7, padding=3)
        self.first = _GNSameBlock(3, block_expansion, kernel_size=7, padding=3, gn=gn)
        down, up = [], []
        for i in range(num_down_blocks):
            down.append(_GNDownBlock(min(max_features, block_expansion * (2 ** i)),
                                     min(max_features, block_expansion * (2 ** (i + 1))), gn=gn))
        for i in range(num_down_blocks):
            up.append(_GNUpBlock(min(max_features, block_expansion * (2 ** (num_down_blocks - i))),
                                 min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1))), gn=gn))
        self.down_blocks = nn.ModuleList(down)
        self.up_blocks = nn.ModuleList(up)
        self.bottleneck = nn.Sequential()
        bc = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), _GNResBlock(bc, gn=gn))
        self.final = nn.Conv2d(block_expansion, 3, 7, padding=3)

    # (1) per-region similarity backward flows: output(driving) grid -> source(cond) grid
    def _sparse_backward(self, coords_t, coords_cond, res, device):
        N = coords_t.shape[0]
        grid = _make_coord_grid(res, res, device).view(1, 1, res, res, 2)
        kp_t = coords_t[..., :2].reshape(N, self.num_kp, 1, 1, 2)
        kp_c = coords_cond[..., :2].reshape(N, self.num_kp, 1, 1, 2)
        s_t = coords_t[..., 2].clamp(min=self.scale_min).reshape(N, self.num_kp, 1, 1, 1)
        s_c = coords_cond[..., 2].clamp(min=self.scale_min).reshape(N, self.num_kp, 1, 1, 1)
        region = kp_c + (s_c / s_t) * (grid - kp_t)
        bg = grid.expand(N, 1, res, res, 2)
        return torch.cat([bg, region], dim=1)

    def _heatmaps(self, coords_t, coords_cond, res, device):
        gt = _region_gaussian(coords_t[..., :2], coords_t[..., 2].clamp(min=self.scale_min), res, device)
        gc = _region_gaussian(coords_cond[..., :2], coords_cond[..., 2].clamp(min=self.scale_min), res, device)
        hm = gt - gc
        bg = torch.zeros(hm.shape[0], 1, res, res, device=device, dtype=hm.dtype)
        return torch.cat([bg, hm], dim=1).unsqueeze(2)

    # (2) dense flow = softmax-mask blend of the K+1 sparse flows (MRAA)
    def _dense_flow(self, cond_img, coords_t, coords_cond):
        N = coords_t.shape[0]; r = self.flow_res; device = cond_img.device
        src = F.interpolate(cond_img, size=(r, r), mode='bilinear', align_corners=False)
        sparse = self._sparse_backward(coords_t, coords_cond, r, device)
        src_rep = src.unsqueeze(1).expand(N, self.num_kp + 1, 3, r, r).reshape(N * (self.num_kp + 1), 3, r, r)
        deformed = F.grid_sample(src_rep, sparse.reshape(N * (self.num_kp + 1), r, r, 2),
                                 align_corners=True, padding_mode='border')
        deformed = deformed.view(N, self.num_kp + 1, 3, r, r)
        hm = self._heatmaps(coords_t, coords_cond, r, device)
        inp = torch.cat([hm, deformed], dim=2).reshape(N, (self.num_kp + 1) * 4, r, r)
        mask = F.softmax(self.mask(self.hourglass(inp)), dim=1)
        flow = (sparse.permute(0, 1, 4, 2, 3) * mask.unsqueeze(2)).sum(1)
        return flow.permute(0, 2, 3, 1)

    # (3) deterministic occlusion via forward splat (Gao), a measurement (no grad)
    @torch.no_grad()
    def _occlusion(self, coords_t, coords_cond):
        N = coords_t.shape[0]; r = self.flow_res; device = coords_t.device
        P = _make_coord_grid(r, r, device).view(1, 1, r, r, 2)
        kp_t = coords_t[..., :2].reshape(N, self.num_kp, 1, 1, 2)
        kp_c = coords_cond[..., :2].reshape(N, self.num_kp, 1, 1, 2)
        s_t = coords_t[..., 2].clamp(min=self.scale_min).reshape(N, self.num_kp, 1, 1, 1)
        s_c = coords_cond[..., 2].clamp(min=self.scale_min).reshape(N, self.num_kp, 1, 1, 1)
        fwd = kp_t + (s_t / s_c) * (P - kp_c)
        d2 = ((P - kp_c) ** 2).sum(-1)
        w = torch.exp(-0.5 * d2 / (s_c.squeeze(-1) ** 2))
        denom = self.bg_resp + w.sum(1, keepdim=True)
        a_k = (w / denom).unsqueeze(-1)
        a_bg = (self.bg_resp / denom).unsqueeze(-1)
        Pxy = P.expand(N, 1, r, r, 2)
        F_flow = (a_bg * Pxy + (a_k * fwd).sum(1, keepdim=True)).squeeze(1)
        fx = (F_flow[..., 0] * 0.5 + 0.5) * (r - 1)
        fy = (F_flow[..., 1] * 0.5 + 0.5) * (r - 1)
        E = self._bilinear_splat(fx, fy, r, N)
        m = ((E > self.occ_lo) & (E < self.occ_hi)).float().unsqueeze(1)
        return m, E

    @staticmethod
    def _bilinear_splat(fx, fy, r, N):
        device = fx.device
        x0 = torch.floor(fx); y0 = torch.floor(fy)
        wx = fx - x0; wy = fy - y0
        x0 = x0.long(); y0 = y0.long(); x1 = x0 + 1; y1 = y0 + 1
        E = torch.zeros(N, r * r, device=device)

        def scat(xi, yi, wgt):
            xi = xi.clamp(0, r - 1); yi = yi.clamp(0, r - 1)
            E.scatter_add_(1, (yi * r + xi).reshape(N, -1), wgt.reshape(N, -1))
        scat(x0, y0, (1 - wx) * (1 - wy)); scat(x1, y0, wx * (1 - wy))
        scat(x0, y1, (1 - wx) * wy);       scat(x1, y1, wx * wy)
        return E.view(N, r, r)

    @staticmethod
    def _deform(inp, flow):
        _, ho, wo, _ = flow.shape
        _, _, h, w = inp.shape
        if ho != h or wo != w:
            flow = F.interpolate(flow.permute(0, 3, 1, 2), size=(h, w), mode='bilinear',
                                 align_corners=False).permute(0, 2, 3, 1)
        return F.grid_sample(inp, flow, align_corners=True, padding_mode='border')

    @staticmethod
    def _gate(warped, prev, occ):
        if occ.shape[2:] != warped.shape[2:]:
            occ = F.interpolate(occ, size=warped.shape[2:], mode='bilinear', align_corners=False)
        if prev is None:
            return warped * occ
        return warped * occ + prev * (1 - occ)

    # (4) generator: MRAA skips + occlusion-gated warp + final source-pixel blend
    def forward(self, cond_img, coords_t, coords_cond):
        flow = self._dense_flow(cond_img, coords_t, coords_cond)
        occ, _ = self._occlusion(coords_t, coords_cond)
        out = self.first(cond_img)
        skips = [out]
        for db in self.down_blocks:
            out = db(out); skips.append(out)
        out = self._gate(self._deform(out, flow), None, occ)
        out = self.bottleneck(out)
        for i, ub in enumerate(self.up_blocks):
            out = self._gate(self._deform(skips[-(i + 1)], flow), out, occ)
            out = ub(out)
        out = self._gate(self._deform(skips[0], flow), out, occ)
        out = torch.sigmoid(self.final(out))
        return self._gate(self._deform(cond_img, flow), out, occ)


class _AntiAliasInterpolation2d(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka
        kernel = 1
        grids = torch.meshgrid([torch.arange(kernel_size, dtype=torch.float32)] * 2, indexing='ij')
        for size, mgrid in zip([kernel_size, kernel_size], grids):
            mean = (size - 1) / 2
            kernel = kernel * torch.exp(-(mgrid - mean) ** 2 / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels; self.scale = scale
        self.int_inv_scale = int(1 / scale)

    def forward(self, x):
        if self.scale == 1.0:
            return x
        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        return out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]


class _ImagePyramide(nn.Module):
    def __init__(self, scales, num_channels=3):
        super().__init__()
        self.scales = list(scales)
        self.downs = nn.ModuleDict({str(s).replace('.', '-'): _AntiAliasInterpolation2d(num_channels, s)
                                    for s in scales})

    def forward(self, x):
        return {'prediction_' + str(s): self.downs[str(s).replace('.', '-')](x) for s in self.scales}


def _load_vgg19_features():
    """Load ImageNet VGG-19 features from PHYWORLD_VGG19_PATH, else timm 'vgg19.tv_in1k' (== torchvision weights)."""
    import os
    from torchvision import models
    net = models.vgg19(weights=None)
    tried = []
    path = os.environ.get('PHYWORLD_VGG19_PATH', '')
    if path and os.path.exists(path):
        try:
            sd = torch.load(path, map_location='cpu', weights_only=False)
            sd = {(k[9:] if k.startswith('features.') else k): v for k, v in sd.items()}
            sd = {k: v for k, v in sd.items() if k in net.features.state_dict()}
            net.features.load_state_dict(sd, strict=True)
            return net.features
        except Exception as e:
            tried.append(f'path={e}')
    try:
        import timm
        m = timm.create_model('vgg19.tv_in1k', pretrained=True)
        sd = {k[9:]: v for k, v in m.state_dict().items() if k.startswith('features.')}
        net.features.load_state_dict(sd, strict=True)
        return net.features
    except Exception as e:
        tried.append(f'timm={e}')
    raise RuntimeError('VGG19 features unavailable (' + ' | '.join(tried) + ')')


class _Vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        f = _load_vgg19_features()
        self.slices = nn.ModuleList()
        for lo, hi in [(0, 2), (2, 7), (7, 12), (12, 21), (21, 30)]:
            s = nn.Sequential()
            for x in range(lo, hi):
                s.add_module(str(x), f[x])
            self.slices.append(s)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        outs = []
        for s in self.slices:
            x = s(x); outs.append(x)
        return outs


class PerceptualPyramidLoss(nn.Module):
    """Multi-scale VGG-19 perceptual loss (FOMM); falls back to multi-scale pixel-L1 if VGG is unavailable."""
    def __init__(self, scales=(1, 0.5, 0.25, 0.125), slice_weights=(1., 1., 1., 1., 1.)):
        super().__init__()
        self.scales = list(scales); self.slice_weights = slice_weights
        self.pyramid = _ImagePyramide(self.scales, 3)
        try:
            self.vgg = _Vgg19(); self.use_vgg = True; self.note = 'vgg19-perceptual'
        except Exception as e:
            self.vgg = None; self.use_vgg = False; self.note = 'MS-L1-fallback:' + str(e)[:100]

    def forward(self, pred, target):
        pred = (pred.clamp(-1, 1) + 1) * 0.5
        target = (target.clamp(-1, 1) + 1) * 0.5
        pp = self.pyramid(pred); pt = self.pyramid(target)
        total = pred.sum() * 0.0
        for s in self.scales:
            a = pp['prediction_' + str(s)]; b = pt['prediction_' + str(s)]
            if self.use_vgg:
                xv = self.vgg(a); yv = self.vgg(b)
                for i, wgt in enumerate(self.slice_weights):
                    total = total + wgt * (xv[i] - yv[i].detach()).abs().mean()
            else:
                total = total + (a - b.detach()).abs().mean()
        return total


class LDR(nn.Module):
    """Encode each frame to a structured latent, roll it forward by kinematic integration, decode by warping the cond frame."""

    def __init__(self, n_kp=16, num_pred=29, width=256, accel_scale=0.5, warp_flow_res=64, kappa_init=0.15):
        super().__init__()
        self.n_kp = n_kp; self.num_pred = num_pred; self.accel_scale = accel_scale
        self.cdim = 3
        # kappa (softplus of log_kappa): uncertainty gate for the measured second-order init
        self.log_kappa = nn.Parameter(torch.log(torch.expm1(torch.tensor(max(float(kappa_init), 1e-3)))))
        self.enc = _KeypointEnc(n_kp)
        d = n_kp * self.cdim
        self.g = nn.Sequential(nn.Linear(2 * d, width), nn.SiLU(),
                               nn.Linear(width, width), nn.SiLU(),
                               nn.Linear(width, d))
        nn.init.zeros_(self.g[-1].weight); nn.init.zeros_(self.g[-1].bias)  # zero-init: rollout starts as pure inertia
        self.warp = _MeasuredWarpRenderer(num_kp=n_kp, flow_res=warp_flow_res)

    def _residual(self, s, v):
        return self.accel_scale * torch.tanh(self.g(torch.cat([s, v], dim=1)))

    def rollout(self, c_cond, n=None):
        n = self.num_pred if n is None else n
        B = c_cond.shape[0]
        assert c_cond.shape[1] >= 3, 'kinematic initialization needs 3 conditioning latents'
        s = c_cond[:, -1].reshape(B, -1)
        v = (c_cond[:, -1] - c_cond[:, -2]).reshape(B, -1)
        a0_raw = (c_cond[:, -1] - 2 * c_cond[:, -2] + c_cond[:, -3]).reshape(B, -1)
        kappa = F.softplus(self.log_kappa)
        a0 = (a0_raw * a0_raw) / (a0_raw * a0_raw + kappa * kappa) * a0_raw
        out = []
        for _ in range(n):
            v = v + a0 + self._residual(s, v)
            s = s + v
            out.append(s)
        return torch.stack(out, 1).view(B, n, self.n_kp, self.cdim)

    def _enc_seq(self, frames):
        B, L = frames.shape[:2]
        return self.enc(frames.reshape(B * L, *frames.shape[2:])).view(B, L, self.n_kp, self.cdim)

    def _decode_seq(self, cond_img, coords_seq, coords_cond):
        B, T = coords_seq.shape[:2]; H, W = cond_img.shape[-2:]
        src01 = ((cond_img.clamp(-1, 1) + 1) * 0.5).unsqueeze(1).expand(B, T, 3, H, W).reshape(B * T, 3, H, W)
        ct = coords_seq.reshape(B * T, self.n_kp, self.cdim)
        cc = coords_cond.unsqueeze(1).expand(B, T, self.n_kp, self.cdim).reshape(B * T, self.n_kp, self.cdim)
        out01 = self.warp(src01, ct, cc)
        return (out01 * 2 - 1).view(B, T, 3, H, W)

    def forward(self, frames, nc, full=False, horizon=None, cond_img=None):
        coords = self._enc_seq(frames)
        coords_cond = coords[:, nc - 1]
        if not full:
            return self._decode_seq(cond_img, self.rollout(coords[:, :nc]), coords_cond)
        dec_ae = self._decode_seq(cond_img, coords, coords_cond)
        roll = self.rollout(coords[:, :nc], horizon)
        dec_roll = self._decode_seq(cond_img, roll, coords_cond)
        return dec_roll, dec_ae, roll, coords


def build_ldr(n_kp=16, num_pred=29, width=256, accel_scale=0.5, warp_flow_res=64, kappa_init=0.15):
    return LDR(n_kp=n_kp, num_pred=num_pred, width=width, accel_scale=accel_scale,
               warp_flow_res=warp_flow_res, kappa_init=kappa_init)
