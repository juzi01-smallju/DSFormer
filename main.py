# -*- coding: utf-8 -*-

import argparse
import os
#  from test import test
# from train import train
from tools.process import Process
from utils import json_type, DotDict
from Net.DSFormer import DSFormer


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--mode', default='train')
    arg.add_argument('--model_name', default='NewNet1')
    arg.add_argument('--preprocess', default=False)
    arg.add_argument('--json_name', type=json_type, default='./config.json')


    arg.add_argument('--in_channel', type=int, default=3,
                     help='the input channel')
    arg.add_argument('--num_classes', type=int, default=3,
                     help='type number')
    arg.add_argument('--init_ratio', type=float, default=2.0,
                     help='reduce-ratio')
    arg.add_argument('--bridge_setting', type=str, default='123',
                     help="M-Bridge's layer seeting, such as ‘0’, '123', '13', '1234'..")

    arg.add_argument('--attention_mechanism', default=None,
                     choices=[None, 'DSSA', 'Traditional'], help="The name of attention_mechanism")

    arg.add_argument('--feedforward', type=str, default='DFFN',
                     choices=['LeFF', 'DFFN', 'Mixffn'], help="The name of Feedforward")

    arg = arg.parse_args()

    # TODO: get config
    config = DotDict(arg.json_name)
    p = Process(config.preprocess)

    net = DSFormer(in_channel=arg.in_channel, )

    if arg.mode == 'train':
        if arg.preprocess:
            p.run_train()
        train(net, arg.model_name, config.train)
