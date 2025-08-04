import numbotics.utils.logger as logger

import numpy as np


def tr(x):
    return np.cos(x) + 1j * np.sin(x)


def tr_inv(x):
    return np.arctan2(x.imag, x.real)


# compute deformation of a wrt b
def deformation(a, b, sample_res=5):
    assert a.shape[0] == b.shape[0]

    smms = a.shape[0]

    D = np.zeros((smms, smms))

    samples = np.arange(0, a.shape[2], sample_res)

    for i in range(smms):
        for j in range(smms):
            diff = 0.0
            for s in samples:
                di = a[i, :, :].T / b[j, :, s]
                di = np.arctan2(di.imag, di.real)
                diff += np.min(np.linalg.norm(di, ord=2, axis=1))
            D[i, j] = diff / samples.shape[0]

    return D


# smm shape: (H, n, S)
# [cclk, clk]
def bound_box(smm):
    # ind 0 is cclk, ind 1 is clk
    ranges = np.zeros((smm.shape[0], smm.shape[1], 2), dtype=np.complex128)
    for i, smm_b in enumerate(smm):

        max_range = np.zeros((smm.shape[1], 2))
        max_range[:, 0] = -np.inf
        max_range[:, 1] = np.inf
        max_val = np.zeros((smm.shape[1], 2), dtype=np.complex128)

        curr_rot = np.zeros((smm.shape[1]))

        for j in range(smm_b.shape[1] - 1):
            diff = tr_inv(smm_b[:, j + 1] / smm_b[:, j])
            curr_rot += diff
            curr_rot = np.where(
                np.abs(curr_rot) >= ((2.0 * np.pi) - 1e-8), np.inf, curr_rot
            )

            cclk_ind = np.where(curr_rot > max_range[:, 0])
            clk_ind = np.where(curr_rot < max_range[:, 1])

            max_range[cclk_ind, 0] = curr_rot[cclk_ind]
            max_range[clk_ind, 1] = curr_rot[clk_ind]

            max_val[cclk_ind, 0] = smm_b[cclk_ind, j + 1]
            max_val[clk_ind, 1] = smm_b[clk_ind, j + 1]

        full_r_ind = np.where(np.abs(curr_rot) > 5.0)
        fixed_r_ind = np.where(np.abs(curr_rot) < 5.0)

        ranges[i, full_r_ind, :] = np.repeat(
            np.array([[-np.inf, np.inf]]), full_r_ind[0].shape[0], axis=0
        )
        ranges[i, fixed_r_ind, :] = max_val[fixed_r_ind, :]

    return ranges


# bb shape: (n, 2)
def intersect_bb(a, b):
    if len(a) == 0 or len(b) == 0:
        return []
    if (a[0].real != -np.inf) and (a[1].real != np.inf):
        assert tr_inv(a[0] / a[1]) != 0.0
    if (b[0].real != -np.inf) and (b[1].real != np.inf):
        assert tr_inv(b[0] / b[1]) != 0.0

    # should probably return these here to avoid doing math on infs
    if a[0].real == -np.inf and a[1].real == np.inf:
        return [[b[0], b[1]]]

    if b[0].real == -np.inf and b[1].real == np.inf:
        return [[a[0], a[1]]]

    a_pi_range = tr_inv(a[0] / a[1]) < 0
    b_pi_range = tr_inv(b[0] / b[1]) < 0

    al_lt_bl = tr_inv(a[0] / b[0]) <= 0.0
    al_gt_bh = tr_inv(a[0] / b[1]) > 0.0
    ah_lt_bl = tr_inv(a[1] / b[0]) < 0.0
    ah_gt_bh = tr_inv(a[1] / b[1]) >= 0.0

    if not (a_pi_range or b_pi_range):
        a_cclk_in_b = al_lt_bl and al_gt_bh
        a_clk_in_b = ah_lt_bl and ah_gt_bh

        b_cclk_in_a = not al_lt_bl and ah_lt_bl
        b_clk_in_a = al_gt_bh and not ah_gt_bh

        if a_cclk_in_b and a_clk_in_b:
            bb_range = [[a[0], a[1]]]
        elif a_cclk_in_b and b_clk_in_a:
            bb_range = [[a[0], b[1]]]
        elif b_cclk_in_a and a_clk_in_b:
            bb_range = [[b[0], a[1]]]
        elif b_cclk_in_a and b_clk_in_a:
            bb_range = [[b[0], b[1]]]
        else:
            bb_range = []

    elif a_pi_range and not b_pi_range:
        b_cclk_in_a = not al_lt_bl or ah_lt_bl
        b_clk_in_a = al_gt_bh or not ah_gt_bh

        if b_cclk_in_a and b_clk_in_a and (ah_lt_bl and al_gt_bh):
            bb_range = [[a[0], b[1]], [b[0], a[1]]]
        elif b_cclk_in_a and b_clk_in_a:
            bb_range = [[b[0], b[1]]]
        elif b_cclk_in_a:
            bb_range = [[b[0], a[1]]]
        elif b_clk_in_a:
            bb_range = [[a[0], b[1]]]
        else:
            bb_range = []

    elif b_pi_range and not a_pi_range:
        a_cclk_in_b = al_lt_bl or al_gt_bh
        a_clk_in_b = ah_lt_bl or ah_gt_bh

        if a_cclk_in_b and a_clk_in_b and (al_gt_bh and ah_lt_bl):
            bb_range = [[a[0], b[1]], [b[0], a[1]]]
        elif a_cclk_in_b and a_clk_in_b:
            bb_range = [[a[0], a[1]]]
        elif a_cclk_in_b:
            bb_range = [[a[0], b[1]]]
        elif a_clk_in_b:
            bb_range = [[b[0], a[1]]]
        else:
            bb_range = []

    else:
        a_cclk_in_b = al_lt_bl or al_gt_bh
        a_clk_in_b = ah_lt_bl or ah_gt_bh

        b_cclk_in_a = not al_lt_bl or ah_lt_bl
        b_clk_in_a = al_gt_bh or not ah_gt_bh

        if (
            (al_lt_bl and ah_lt_bl)
            or (al_gt_bh and ah_gt_bh)
            or (al_gt_bh and ah_lt_bl)
        ):
            bb_range = [[a[0], b[1]], [b[0], a[1]]]
        elif a_cclk_in_b and not a_clk_in_b:
            bb_range = [[a[0], b[1]]]
        elif a_clk_in_b and not a_cclk_in_b:
            bb_range = [[b[0], a[1]]]
        elif a_cclk_in_b and a_clk_in_b:
            bb_range = [[a[0], a[1]]]
        elif b_cclk_in_a and b_clk_in_a:
            bb_range = [[b[0], b[1]]]
        else:
            bb_range = []

    return bb_range


def smm_bb_intersect(smm_a, smm_b):
    assert smm_a.shape[1] == smm_b.shape[1]

    bb_a = bound_box(smm_a)
    bb_b = bound_box(smm_b)

    limits = np.empty((bb_a.shape[0], bb_b.shape[0]), dtype=object)

    for i in range(bb_a.shape[0]):
        for j in range(bb_b.shape[0]):

            ai_bj_lims = []
            for n in range(bb_a.shape[1]):
                ai_bj_lims.append(intersect_bb(bb_a[i, n, :], bb_b[j, n, :]))

            limits[i, j] = ai_bj_lims

    return limits


# magic
def bb_to_lin_eq(bb):
    lims = np.array(bb)
    ws = np.empty(lims.shape, dtype=float)
    bs = np.empty(lims.shape[:2], dtype=float)
    n = lims.shape[1]

    offsets = tr(np.pi - 1e-6) / lims[:, :, 0]
    a = lims[:, :, 0] * offsets
    b = lims[:, :, 1] * offsets
    c = (np.sqrt(a)) * (np.sqrt(b))
    c = c / offsets
    b = (lims[:, :, 0] + lims[:, :, 1]) / 2.0
    w = b - c
    w = np.concatenate((w[..., np.newaxis].real, w[..., np.newaxis].imag), axis=-1)
    w = w / (np.linalg.norm(w, ord=2, axis=-1)[..., np.newaxis])
    b = (b.real * w[:, :, 0]) + (b.imag * w[:, :, 1])

    w = np.where(np.isnan(w), np.inf, w)
    w = np.where(np.isinf(w), 0.0, w)
    b = np.where(np.isnan(b), np.inf, b)

    return w, b


# magic
def dist_from_bb(bb, w, b, qs, signed=False, bb_index=False, weights=None):

    lims = np.array(bb)
    B = lims.shape[0]

    qs_vec = np.stack((qs.real, qs.imag), -1)
    qs_vec = np.tile(qs_vec[:, np.newaxis], (1, B, 1, 1))

    q_dot_w = np.sum(qs_vec[..., :] * w[..., :], axis=-1)
    qw_minus_b = q_dot_w - b
    qw_minus_b = np.stack(
        (qw_minus_b[..., np.newaxis], qw_minus_b[..., np.newaxis]), -1
    )

    qw_minus_b = np.reshape(qw_minus_b, (-1,))

    bb_diffs = tr_inv(lims / np.tile(qs[:, np.newaxis, :, np.newaxis], (1, B, 1, 2)))
    bb_diffs[np.where(np.isnan(bb_diffs))] = 0.0
    bb_diffs[np.where(np.isinf(bb_diffs))] = 0.0

    b_shape = bb_diffs.shape
    bb_diffs = np.reshape(bb_diffs, (-1,))
    in_bb = np.where(qw_minus_b < 0)
    bb_diffs[in_bb] = 0.0
    bb_diffs = np.reshape(bb_diffs, b_shape)

    if signed:
        signed_inds = np.argmin(np.abs(bb_diffs), axis=-1)
        signed_diffs = np.squeeze(
            np.take_along_axis(bb_diffs, signed_inds[..., None], axis=-1), axis=-1
        )

        which_diff = np.argmin(np.linalg.norm(signed_diffs, axis=-1), axis=1)[
            :, np.newaxis, np.newaxis
        ]
        which_inds = np.broadcast_to(
            which_diff, (signed_diffs.shape[0], 1, signed_diffs.shape[2])
        )
        signed_diffs = np.take_along_axis(signed_diffs, which_inds, axis=1)[:, 0, :]

    bb_diffs = np.amin(np.abs(bb_diffs), axis=-1)

    if weights is not None:
        assert (weights.shape == (bb_diffs.shape[-1],)) or (
            weights.shape == (bb_diffs.shape[0], bb_diffs.shape[-1])
        )
        if weights.shape == (bb_diffs.shape[-1],):
            bb_diffs *= weights[np.newaxis, np.newaxis, :]
        else:
            bb_diffs *= weights[:, np.newaxis, :]

    bb_norm = np.linalg.norm(bb_diffs, ord=2, axis=-1)
    if bb_index:
        bb_inds = np.argmin(bb_norm, axis=-1)

    min_indices = np.argmin(
        np.tile(bb_norm[..., np.newaxis], (1, 1, bb_diffs.shape[-1])), axis=1
    )

    bb_diffs = np.take_along_axis(
        bb_diffs, np.expand_dims(min_indices, axis=1), axis=1
    ).squeeze(1)

    if signed and bb_index:
        return bb_diffs, signed_diffs, bb_inds
    elif bb_index:
        return bb_diffs, bb_index
    elif signed:
        return bb_diffs, signed_diffs
    return bb_diffs
