import sys
import os
import argparse
import pickle as pkl
import torch
import numpy as np

# add pytorch vqa to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VQA_DIR = os.path.join(CURRENT_DIR, 'pytorch-vqa')
sys.path.append(VQA_DIR)

import model
import data
import utils
from train import run


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str, default=os.path.join(CURRENT_DIR, 'checkpoints', '2017-08-04_00.55.19.pth'))


def main(args):
    # load pre-trained model
    log = torch.load(args.ckpt_path)
    tokens = len(log['vocab']['question']) + 1
    net = torch.nn.DataParallel(model.Net(tokens))
    net.load_state_dict(log['weights'])

    # create val loader
    val_loader = data.get_loader(val=True)
    tracker = utils.Tracker()
    r = run(net, val_loader, None, tracker, train=False, prefix='val', epoch=50)
    #print('r:', r)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

