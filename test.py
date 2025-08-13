# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataload.loader import EyeBallData
from utils import restore_vessel


def test(config, net, model_path):

    model = net

    model_name = model_path.split("/")[-1].split('.')[0]

    print(f'pth name: {model_name}')

    model.load_state_dict(torch.load(model_path,
                                     map_location={'cuda:1': 'cuda:3', 'cuda:0': 'cuda:3', 'cuda:2': 'cuda:3'}), False)
    model = model.to(config.device)

    testLoad = EyeBallData(config, 'test')
    testLoader = DataLoader(testLoad, shuffle=False)

    os.makedirs(f'./Result/{model_name}', exist_ok=True)

    with tqdm(total=len(testLoader), desc='Test', ncols=80) as bar:
        model.eval()
        for item, (data, originalSize, currentSize, name) in enumerate(testLoader):
            resImgDict = data['img']

            originalSize = tuple(map(int, originalSize))
            currentSize = tuple(map(int, currentSize))

            res = {}
            for k in resImgDict.keys():
                outListAv = []
                for idx, imgAv in enumerate(resImgDict[k]):
                    with torch.no_grad():
                        imgAv = imgAv.to(config.device)
                        imgAv = Variable(imgAv)

                        predAv = model(imgAv)
                        _, predAv = torch.max(predAv.cpu().data, 1)
                        predAv = predAv[0]

                        outListAv.append(predAv)

                res[k] = outListAv

            outAv = restore_vessel(threshold_vessel(res, currentSize))

            saveAv = Image.fromarray(np.uint8(cv2.resize(outAv, originalSize) * 255)).convert('RGB')

            saveAv.save(os.path.join(f'./Result/{modelName}', name[0]))
            bar.update(1)


def threshold_vessel(res, currentSize):
    w, h = currentSize

    s = 256

    m = w // s + 1
    n = h // s + 1

    size = (n * s, m * s)

    outAv = []
    for k in res.keys():

        newAvImg = np.zeros(size)

        for i, out in enumerate(res[k]):
            newAvImg[int(i % n) * s: int(i % n) * s + s, int(i / n) * s: int(i / n) * s + s] = out

        outAv.append(newAvImg[k[0]: k[0] + h, k[1]: k[1] + w])

    av = np.zeros_like(outAv[0])
    for i in range(len(outAv[0])):
        for j in range(len(outAv[0][0])):
            pointSum = [0, 0, 0]
            for tmp in outAv:
                try:
                    pointSum[int(tmp[i][j])] += 1
                except:
                    raise
            av[i][j] = pointSum.index(max(pointSum))

    return av
