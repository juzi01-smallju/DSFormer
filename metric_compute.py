# !/usr/bin/env python
# -*-coding:utf-8 -*-

import os

import cv2
import numpy as np


def compute_metric(t, p):
    # t = t / 255
    t = np.where(t / 255 > 0.5, 1, 0)
    p = np.where(p / 255 > 0.5, 1, 0)

    tpp = ((t[:, :, 2] == 1) & (p[:, :, 2] == 1)).sum()
    tnp = ((t[:, :, 0] == 1) & (p[:, :, 0] == 1)).sum()
    fpp = ((t[:, :, 0] == 1) & (p[:, :, 2] == 1)).sum()
    fnp = ((t[:, :, 2] == 1) & (p[:, :, 0] == 1)).sum()

    return tpp, tnp, fpp, fnp


# print(acc, precision, recall)
tp, tn, fp, fn = 0, 0, 0, 0
for image in os.listdir('data/LES/256/validation/label'):
    target = cv2.imread('./data/LES/256/validation/label/' + image)
    predict = cv2.imread('./Result/DRIVE/39/' + image)

    # 计算预测结果和真实结果的准确率、精确率、召回率
    ttp, ttn, tfp, tfn = compute_metric(target, predict)
    tp, tn, fp, fn = tp + ttp, tn + ttn, fp + tfp, fn + tfn

acc = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f'准确率、精确率、召回率为：{acc}, {precision}, {recall}')
# t = cv2.imread('./data/test/label/DRIVE_0.png')
# p = cv2.imread('./Result/24_epoch_0.907/DRIVE_0.png')
