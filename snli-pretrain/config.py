import os

# paths
base_dir   = os.path.expanduser('~')
data_dir   = os.path.join(base_dir, 'data')
snli_path  = os.path.join(data_dir, 'SNLI')
flickr_path = os.path.join(data_dir, 'flickr30k_images/flickr30k_images')
vocabulary_path   = os.path.join(data_dir, 'snli-vocab.json')  # path where the used vocabularies for question and answers are saved to
generated_images_path = os.path.join(
    base_dir,
    '/github/text_to_image_qa/vqa/AttnGAN/models/coco_AttnGAN2/train/single')

task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 3  # can probably do 6
image_size            = 299  # inception image size
# image_size            = 448  # scale shorter end of image to this size and centre crop
output_size           = image_size // 32  # size of the feature maps after processing through a network
output_features       = 2048  # number of feature maps thereof
central_fraction      = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs       = 10
batch_size   = 128
initial_lr    = 1e-3  # default Adam lr
lr_halflife  = 50000  # in iterations
data_workers = 8
max_answers  = 3

# real images vs. generated
use_generated_images = False
