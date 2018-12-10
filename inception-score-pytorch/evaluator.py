import numpy as np
import glob
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from inception_score import inception_score


class RunningStats(object):

    def __init__(self):
        '''
        Returns instance of stats reader
        '''
        self.num_obs = 0
        self.mean = None
        self.std = None

    def update(self, new_mean, new_std, n):
        '''
        Returns the running mean and std for the new batch mean and std
        '''
        if self.num_obs == 0:
            self.mean = new_mean
            self.std  = new_std
            self.num_obs = n
        else:
            # make float
            m = self.num_obs * 1.0
            tmp = self.mean
            # update running mean std
            self.mean = m/(m+n)*tmp + n/(m+n)*new_mean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*new_std**2 +\
                        m*n/(m+n)**2 * (tmp - new_mean)**2
            self.std  = np.sqrt(self.std)
            # increment num obs
            self.num_obs += n
        return self.mean, self.std


class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


class InceptionScoreEvaluator(object):

    def __init__(self, image_dir):
	self.image_dir = image_dir

    def evaluate(self):
        # TODO: maybe we shouldn't normalize
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	data = datasets.ImageFolder(self.image_dir, transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), normalize]))
        mean_score, std_score = inception_score(IgnoreLabelDataset(data))
        return mean_score, std_score


if __name__ == '__main__':
    # generated images as list of numpy arrays
    start = datetime.now()
    image_dir_list = [
        #'../finetune-attngan/models/coco_AttnGAN2_120/valid/',
        #'../finetune-attngan/output/coco_glu-gan2_2018_12_03_04_42_22/Model/netG_epoch_121/valid/',
        '../finetune-attngan/output/coco_glu-gan2_2018_12_08_23_05_29/Model/netG_epoch_56/valid/'
        #'../finetune-attngan/output/coco_glu-gan2_2018_12_07_04_55_13/Model/netG_epoch_56/valid/'
    ]
    for image_dir in image_dir_list:
        evaluator = InceptionScoreEvaluator(image_dir)
        mean, std = evaluator.evaluate()
        print('image dir:', image_dir, 'has mean:', mean, ' and std:', std)
        print('time to eval:', datetime.now()-start)






