### CONFIGS ###
dataset = 'cifar10'
model = 'AE'

channels = 3
dim = 64
latent_didm = 100

num_epoch = 200
learning_rate = 0.01

dataroot = '/media/ouc/4T_A/datasets/cifar10'
save_image_path = 'checkpoints/ae_geo'
save_model_path = 'models/ae_geo'
log_txt = 'logs/ae_geo'
image_size = 64
batch_size = 8
mode = 'train'
num_workers = 1

input_crop_size = 64
cur_iter = 0
iter_for_max_range = 10000
must_divide = 8
min_scale = 0.15
max_scale = 2.25
max_transform_magnitude = 0.0