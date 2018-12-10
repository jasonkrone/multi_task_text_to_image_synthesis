#!/bin/bash

# download anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh

# create env for pytorch vqa model
conda create -n vqa python=3.6.2
conda install pytorch torchvision -c pytorch
source activate vqa
pip install -r vqa_requirements.txt
source deactivate

# create env for gan model
conda create -n gan python=2.7
source activate gan
pip install -r gan_requirements.txt
conda install pytorch torchvision -c pytorch

# download the checkpoint for vqa on vqa_v1
wget -P ./checkpoints https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth 

# download the checkpoint for gan on coco
wget -P ./checkpoints https://drive.google.com/a/columbia.edu/uc?authuser=0&id=1i9Xkg9nU74RAvkcqKE-rJYhjvzKAMnCi&export=download

# TODO: gpu versions, gan expects py2, vqa expects py3

# download data
mkdir ~/data/vqa
gsutil cp gs://calberti-vision-bucket/data/vqa_v1/test2015.zip ~/data/.
gsutil cp gs://calberti-vision-bucket/data/vqa_v1/train2014.zip ~/data/. 
gsutil cp gs://calberti-vision-bucket/data/vqa_v1/val2014.zip ~/data/. 
gsutil cp gs://calberti-vision-bucket/data/vqa_v1/train2014-generated.zip ~/data/.
gsutil cp gs://calberti-vision-bucket/data/vqa_v1/val-generated.zip ~/data/.
gsutil cp -R gs://calberti-vision-bucket/data/vqa_v1/jsons/ ~/data/.

# unzip
unzip ~/data/'*.zip' -d ~/data/.
