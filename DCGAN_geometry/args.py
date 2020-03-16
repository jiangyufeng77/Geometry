### CONFIGS ###
dataset = 'cifar10'
model = 'AE'

channels = 3
dim = 64
latent_didm = 100

num_epoch = 200
learning_rate = 0.01

dataroot = '/media/ouc/4T_A/datasets/cifar10'
save_image_path = 'checkpoints/ae_ori'
save_model_path = 'models/ae_ori'
log_txt = 'logs/ae_ori'
image_size = 64
batch_size = 8
mode = 'train'
num_workers = 1