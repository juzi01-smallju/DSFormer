# !/usr/bin/env python
# -*-coding:utf-8 -*-

import argparse
import json

import numpy as np


class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return DotDict(value)
        return value


def json_type(s):
    try:
        return json.load(open(s, 'r'))
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_prediction(out_list, current_size, data_mode):
    # out_list shape = [number, 3, H, W] [number, H, W]
    w, h = current_size  # 500, 500
    s = out_list[0].shape[1]  # 256

    m = w // s + (0 if not w % s else 1)  # 2
    n = h // s + (0 if not h % s else 1)  # 2

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    if data_mode == 'multi':
        pre_av = np.zeros((n * s, m * s))
        for j in range(m):
            for i in range(n):
                pre_av[i * s: (i + 1) * s, j * s: (j + 1) * s] = out_list[j * n + i]
        pre_av = pre_av[tmp_h: tmp_h + h, tmp_w: tmp_w + w]

    else:
        pre_av = np.zeros((3, n * s, m * s))
        for j in range(m):
            for i in range(n):
                pre_av[:, i * s: (i + 1) * s, j * s: (j + 1) * s] = out_list[j * n + i]
        pre_av = pre_av[:, tmp_h: tmp_h + h, tmp_w: tmp_w + w]

    return pre_av  # 3, H, W


def restore(data):
    r, g, b = np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)

    r[data == 3] = 1
    g[data == 1] = 1
    b[data == 2] = 1

    res = np.zeros((*data.shape, 3))

    res[:, :, 0] = r
    res[:, :, 2] = b
    res[:, :, 1] = g

    return res

