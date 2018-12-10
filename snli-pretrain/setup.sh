#!/bin/bash

# setup anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
conda create -n snli python=2.7.13
conda install pytorch torchvision -c pytorch -n snli

# download data
mkdir ~/data/
mkdir ./logs
gsutil cp gs://calberti-vision-bucket/data/flickr30k/flickr30k_images.zip ~/data/
conda activate snli

# install dependencies
pip install --user -r requirements.txt
pip install --user tqdm
pip install --user scipy
python download_glue_data.py --data_dir ~/data --tasks SNLI
python preprocess-vocab.py

# # unzip
unzip ~/data/'*.zip' -d ~/data/.
