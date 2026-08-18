"""
Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.

This file constitutes Licensed Materials under the Adobe Research License.
Use is limited to noncommercial research purposes.
See the LICENSE file at the project root for the complete license terms and disclaimer.

PhyWorld white-box parser: extract each ball's center/radius from predicted frames and score vs GT.
"""

import numpy as np
import pandas as pd

left_color = (255, 0, 0)
right_color = (0, 0, 255)
COLORS = [left_color, right_color]
WORLD_SCALE = 10.0


def parse_state_from_image(image_rgb, default_y, default_r1, thres=0.15, color=0):
    image_copy = image_rgb.copy()
    all_scaled_circles = []
    if color == 0:
        circle_mask = (image_copy[:, :, 0] > 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] < 127)
    elif color == 1:
        circle_mask = (image_copy[:, :, 0] < 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] > 127)
    else:
        raise ValueError("Invalid color")

    area = np.sum(circle_mask)
    radius = np.sqrt(area / np.pi)
    scaled_radius = radius / image_copy.shape[1] * WORLD_SCALE
    if area > thres:
        center = (
            np.mean(np.nonzero(circle_mask)[1]),
            np.mean(np.nonzero(circle_mask)[0]),
        )
        scaled_center = (
            center[0] / image_copy.shape[1] * WORLD_SCALE,
            center[1] / image_copy.shape[1] * WORLD_SCALE,
        )
    else:
        scaled_center = (WORLD_SCALE, default_y)
        scaled_radius = default_r1

    all_scaled_circles.append([*scaled_center, scaled_radius])
    return np.array(all_scaled_circles)


def parse_state_from_image_collision(image_rgb, default_y, default_r1, default_r2, thres=0.15):
    image_copy = image_rgb.copy()
    all_scaled_circles = []
    for ball_id, color in enumerate(COLORS):
        if color == (255, 0, 0):
            circle_mask = (image_copy[:, :, 0] > 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] < 127)
        elif color == (0, 0, 255):
            circle_mask = (image_copy[:, :, 0] < 127) & (image_copy[:, :, 1] < 127) & (image_copy[:, :, 2] > 127)
        else:
            raise ValueError("Invalid color")

        area = np.sum(circle_mask)
        radius = np.sqrt(area / np.pi)
        scaled_radius = radius / image_copy.shape[1] * WORLD_SCALE
        if area > thres:
            center = (
                np.mean(np.nonzero(circle_mask)[1]),
                np.mean(np.nonzero(circle_mask)[0]),
            )
            scaled_center = (
                center[0] / image_copy.shape[1] * WORLD_SCALE,
                center[1] / image_copy.shape[1] * WORLD_SCALE,
            )
        else:
            if ball_id == 0:
                scaled_center = (0, default_y)
                scaled_radius = default_r1
            else:
                scaled_center = (WORLD_SCALE, default_y)
                scaled_radius = default_r2

        all_scaled_circles.append([*scaled_center, scaled_radius])
    return np.array(all_scaled_circles)


def get_last_ema(values_list, span):
    series = pd.Series(values_list)
    ema = series.ewm(span=span, adjust=False).mean()
    return ema.iloc[-1]


def xy_metrics(list_a, list_b):
    distances_x, distances_y = [], []
    assert len(list_a) == len(list_b) == 1
    for elem_a, elem_b in zip(list_a, list_b):
        if not np.any(np.isnan(elem_a)) and not np.any(np.isnan(elem_b)):
            distances_x.append(np.abs(elem_a[0] - elem_b[0]))
            distances_y.append(np.abs(elem_a[1] - elem_b[1]))
    x_error_avg = np.mean(distances_x) if distances_x else np.nan
    y_error_avg = np.mean(distances_y) if distances_y else np.nan
    return x_error_avg, y_error_avg


def evaluate_xy(rollout_frames, gt_features, init, mode, gamma=0.98, sample_freq=1, pred_states=None):
    assert sample_freq == 1, 'there may be some bugs if it is greater than 1'

    # keep only frames where the ball is fully in view
    left_ball_r = init[0]
    left_ball_init_v = init[1]
    left_ball_m = init[0]**2
    index = []
    CONDITION_FRAMES = 4
    for i, state in enumerate(gt_features):
        if i < CONDITION_FRAMES-1:
            continue
        left_ball_x, left_ball_y = state[0], state[1]
        if left_ball_x - left_ball_r >= 0 and left_ball_x + left_ball_r <= WORLD_SCALE \
            and left_ball_y - left_ball_r >= 0 and left_ball_y + left_ball_r <= WORLD_SCALE:
            index.append(i)

    if rollout_frames is not None:
        assert len(rollout_frames) == len(gt_features), f'{len(rollout_frames)}, {len(gt_features)}'
        rollout_frames = rollout_frames[index]
        gt_features = gt_features[index]

        x_error_list, y_error_list, r_list = [], [], []
        x_pos_list, y_pos_list = [], []
        default_y, default_r1 = np.nan, np.nan
        for rollout_frame, gt_feature in zip(rollout_frames[sample_freq-1::sample_freq], gt_features[sample_freq-1::sample_freq]):
            parsed_state = parse_state_from_image(rollout_frame, default_y, default_r1, color=init[2] if len(init) >= 3 else 0)
            default_y, default_r1 = parsed_state[0][1], parsed_state[0][2]
            x_pos_list.append(parsed_state[:, 0])
            y_pos_list.append(parsed_state[:, 1])
            r_list.append(parsed_state[:, 2])
            parsed_state = parsed_state[:, :2]
            x_error, y_error = xy_metrics([gt_feature], parsed_state)
            x_error_list.append(x_error)
            y_error_list.append(y_error)
        span = len(rollout_frames[sample_freq-1::sample_freq])
        ema_x_error = get_last_ema(x_error_list, span=span)
        ema_y_error = get_last_ema(y_error_list, span=span)

    else:
        assert pred_states is not None and len(pred_states) == len(gt_features), f'{len(pred_states)}, {len(gt_features)}'
        pred_states = pred_states[index]
        gt_features = gt_features[index]
        x_pos_list = pred_states[:, 0]
        y_pos_list = pred_states[:, 1]
        r_list = pred_states[:, 2]

    r1_list = r_list
    r1_min, r1_max = min(r1_list), max(r1_list)
    delta_r = r1_max - r1_min

    gt_x_pos = gt_features[:, 0]
    gt_x_vel = np.diff(gt_x_pos, axis=0) / (0.1 * sample_freq)
    gt_y_pos = gt_features[:, 1]
    gt_y_vel = np.diff(gt_y_pos, axis=0) / (0.1 * sample_freq)

    x_pos = np.array(x_pos_list)
    y_pos = np.array(y_pos_list)
    if x_pos.shape[-1] == 1:
        x_pos = np.squeeze(x_pos, -1)
        y_pos = np.squeeze(y_pos, -1)

    x_vel = np.diff(x_pos, axis=0) / (0.1 * sample_freq)
    y_vel = np.diff(y_pos, axis=0) / (0.1 * sample_freq)

    x_err_avg = np.mean(np.abs(x_pos - gt_x_pos))
    y_err_avg = np.mean(np.abs(y_pos - gt_y_pos))

    x_vel_err_avg = np.mean(np.abs(x_vel - gt_x_vel))
    y_vel_err_avg = np.mean(np.abs(y_vel - gt_y_vel))

    if mode == 'all':
        return {
            'init': init,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'x_vel': x_vel,
            'y_vel': y_vel,
            'r': r_list,
            'gt_x_pos': gt_x_pos,
            'gt_y_pos': gt_y_pos,
            'abs_x_err_avg': x_err_avg,
            'abs_y_err_avg': y_err_avg,
            'delta_r': delta_r,
            'abs_x_vel_err_avg': x_vel_err_avg,
            'abs_y_vel_err_avg': y_vel_err_avg,
        }
    else:
        return ema_x_error, ema_y_error


def evaluate_xy_collision(rollout_frames, gt_features, init, mode, gamma=0.98, sample_freq=1, pred_states=None):
    assert sample_freq == 1, 'there may be some bugs if it is greater than 1'

    left_ball_r, right_ball_r = init[:2]
    left_ball_init_v, right_ball_init_v = init[2:4]
    left_ball_m, right_ball_m = init[:2]**2
    index = []
    CONDITION_FRAMES = 4
    for i, state in enumerate(gt_features):
        if i < CONDITION_FRAMES-1:
            continue
        left_ball_x = state[0, 0]
        right_ball_x = state[1, 0]
        if left_ball_x - left_ball_r >= 0 and right_ball_x + right_ball_r <= WORLD_SCALE:
            index.append(i)

    if rollout_frames is not None:
        assert len(rollout_frames) == len(gt_features), f'{len(rollout_frames)}, {len(gt_features)}'
        rollout_frames = rollout_frames[index]
        gt_features = gt_features[index]

        r_list = []
        x_pos_list = []
        y_pos_list = []
        default_y, default_r1, default_r2 = np.nan, np.nan, np.nan
        for rollout_frame, gt_feature in zip(rollout_frames[sample_freq-1::sample_freq], gt_features[sample_freq-1::sample_freq]):
            parsed_state = parse_state_from_image_collision(rollout_frame, default_y, default_r1, default_r2)
            default_y, default_r1, default_r2 = parsed_state[0][1], parsed_state[0][2], parsed_state[1][2]
            x_pos_list.append(parsed_state[:, 0])
            y_pos_list.append(parsed_state[:, 1])
            r_list.append(parsed_state[:, 2])
            parsed_state = parsed_state[:, :2]

    else:
        assert pred_states is not None and len(pred_states) == len(gt_features), f'{len(pred_states)}, {len(gt_features)}'
        pred_states = pred_states[index]
        gt_features = gt_features[index]
        x_pos_list = pred_states[:, :2]
        y_pos_list = pred_states[:, 2:4]
        r_list = pred_states[:, 4:]

    # only use frames before collision to avoid out-of-vision region
    MIN_POST_FRAMES = 8
    r_list = r_list[:(MIN_POST_FRAMES-CONDITION_FRAMES+1)//sample_freq]
    r1_list = [x[0] for x in r_list]
    r2_list = [x[1] for x in r_list]
    r1_min, r1_max = min(r1_list), max(r1_list)
    r2_min, r2_max = min(r2_list), max(r2_list)
    delta_r = max(r1_max - r1_min, r2_max - r2_min)

    gt_x_pos = gt_features[:, :, 0]
    gt_y_pos = gt_features[:, :, 1]
    gt_x_vel = np.diff(gt_x_pos, axis=0) / (0.1 * sample_freq)

    # collision frame = first large change in GT x-velocity
    collision_index = len(gt_x_vel) - 1
    for i in range(1, len(gt_x_vel)):
        if np.abs(gt_x_vel[i, 0] - gt_x_vel[i-1, 0]) > 0.1 or np.abs(gt_x_vel[i, 1] - gt_x_vel[i-1, 1]) > 0.1:
            collision_index = i
            break

    x_pos = np.array(x_pos_list)
    x_vel = np.diff(x_pos, axis=0) / (0.1 * sample_freq)

    pre_x_err_avg = np.mean(np.abs(x_pos[:collision_index+1] - gt_x_pos[:collision_index+1]))
    post_x_err_avg = np.mean(np.abs(x_pos[collision_index+1:] - gt_x_pos[collision_index+1:]))

    y_pos = np.array(y_pos_list)
    pre_y_err_avg = np.mean(np.abs(y_pos[:collision_index+1] - gt_y_pos[:collision_index+1]))
    post_y_err_avg = np.mean(np.abs(y_pos[collision_index+1:] - gt_y_pos[collision_index+1:]))

    pre_vel_err_avg = np.mean(np.abs(x_vel[:collision_index] - gt_x_vel[:collision_index]))
    post_vel_err_avg = np.mean(np.abs(x_vel[collision_index+1:] - gt_x_vel[collision_index+1:]))

    momentum = left_ball_m * x_vel[:, 0] + right_ball_m * x_vel[:, 1]
    gt_momentum = left_ball_m * left_ball_init_v - right_ball_m * right_ball_init_v
    pre_momentum_error_avg = np.mean(np.abs(momentum[:collision_index] - gt_momentum))
    post_momentum_error_avg = np.mean(np.abs(momentum[collision_index+1:] - gt_momentum))

    energy = left_ball_m * x_vel[:, 0]**2 / 2 + right_ball_m * x_vel[:, 1]**2 / 2
    gt_energy = left_ball_m * left_ball_init_v**2 / 2 + right_ball_m * right_ball_init_v**2 / 2
    pre_energy_error_avg = np.mean(np.abs(energy[:collision_index] - gt_energy))
    post_energy_error_avg = np.mean(np.abs(energy[collision_index+1:] - gt_energy))

    return {
        'init': init,
        'collision_index': int(collision_index),
        'pre_x_vel': x_vel[:collision_index],
        'gt_pre_x_vel': gt_x_vel[:collision_index],
        'post_x_vel': x_vel[collision_index+1:],
        'gt_post_x_vel': gt_x_vel[collision_index+1:],
        'x_pos': x_pos,
        'gt_x_pos': gt_x_pos,
        'x_vel': x_vel,
        'abs_x_err_avg': float(np.mean(np.abs(x_pos - gt_x_pos))),
        'abs_y_err_avg': float(np.mean(np.abs(y_pos - gt_y_pos))),
        'y_pos': y_pos,
        'gt_y_pos': gt_y_pos,
        'delta_r': delta_r,
        'pre_x_err_avg': pre_x_err_avg,
        'post_x_err_avg': post_x_err_avg,
        'pre_y_err_avg': pre_y_err_avg,
        'post_y_err_avg': post_y_err_avg,
        'pre_vel_err_avg': pre_vel_err_avg,
        'post_vel_err_avg': post_vel_err_avg,
        'pre_momentum_error_avg': pre_momentum_error_avg,
        'post_momentum_error_avg': post_momentum_error_avg,
        'pre_energy_error_avg': pre_energy_error_avg,
        'post_energy_error_avg': post_energy_error_avg,
    }


def evaluate_xy_looming(rollout_frames, gt_pos, gt_radius, init, mode='all', sample_freq=1):
    """Looming (scale dynamics): parsed ball radius r(t) vs GT radius, plus x/y position; primary axis = radius."""
    CONDITION_FRAMES = 4
    m = min(len(rollout_frames), len(gt_pos), len(gt_radius))
    idx = []
    for i in range(m):
        if i < CONDITION_FRAMES - 1:
            continue
        x, y, r = float(gt_pos[i, 0]), float(gt_pos[i, 1]), float(gt_radius[i])
        if x - r >= 0 and x + r <= WORLD_SCALE and y - r >= 0 and y + r <= WORLD_SCALE:
            idx.append(i)
    if len(idx) == 0:
        return {'abs_r_err_avg': np.nan, 'abs_x_err_avg': np.nan, 'abs_y_err_avg': np.nan, 'abs_r_vel_err_avg': np.nan}
    xs, ys, rs = [], [], []
    default_y, default_r = np.nan, np.nan
    for i in idx:
        ps = parse_state_from_image(rollout_frames[i], default_y, default_r, color=0)
        default_y, default_r = ps[0][1], ps[0][2]
        xs.append(float(ps[0][0])); ys.append(float(ps[0][1])); rs.append(float(ps[0][2]))
    xs, ys, rs = np.array(xs), np.array(ys), np.array(rs)
    gx = np.array([float(gt_pos[i, 0]) for i in idx]); gy = np.array([float(gt_pos[i, 1]) for i in idx])
    gr = np.array([float(gt_radius[i]) for i in idx])
    r_vel_err = float(np.mean(np.abs(np.diff(rs) - np.diff(gr)) / (0.1 * sample_freq))) if len(rs) > 1 else np.nan
    return {
        'init': init, 'r': rs, 'gt_r': gr, 'x_pos': xs, 'y_pos': ys,
        'abs_r_err_avg': float(np.mean(np.abs(rs - gr))),
        'abs_x_err_avg': float(np.mean(np.abs(xs - gx))),
        'abs_y_err_avg': float(np.mean(np.abs(ys - gy))),
        'abs_r_vel_err_avg': r_vel_err,
    }
