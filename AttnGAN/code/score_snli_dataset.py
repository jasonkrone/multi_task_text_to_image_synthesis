from __future__ import print_function
from datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from model import RNN_ENCODER, CNN_ENCODER
import pretrain_DAMSM

import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--output_scores', dest='output_scores', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
  args = parse_args()
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

  if args.gpu_id == -1:
    cfg.CUDA = False
  else:
    cfg.GPU_ID = args.gpu_id

  if args.data_dir != '':
    cfg.DATA_DIR = args.data_dir

  imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
  image_transform = transforms.Compose([
      transforms.Scale(int(imsize * 76 / 64)),
      transforms.RandomCrop(imsize),
      transforms.RandomHorizontalFlip()])
  split_dir = 'test'
  dataset = TextDataset(cfg.DATA_DIR, split_dir,
                        base_size=cfg.TREE.BASE_SIZE,
                        transform=image_transform)

  batch_size = cfg.TRAIN.BATCH_SIZE
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, drop_last=True,
      shuffle=True, num_workers=int(cfg.WORKERS))

  torch.cuda.set_device(cfg.GPU_ID)
  cudnn.benchmark = True
  text_encoder, image_encoder, labels, start_epoch = (
      pretrain_DAMSM.build_models(dataset, batch_size))

  s_loss, w_loss, similarities_by_key = pretrain_DAMSM.evaluate(
      dataloader, image_encoder, text_encoder, batch_size, labels)

  with file(args.output_scores, 'w') as fout:
    fout.write(
        "\n".join("%s\t%f" % it for it in sorted(similarities_by_key.items())))

  print("s_loss, w_loss:", (s_loss, w_loss))
