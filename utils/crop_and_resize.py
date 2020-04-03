import random
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

path = '/Users/jiangyufeng/Desktop/草莓/'
path2 = '/Users/jiangyufeng/Desktop/草莓resize/'
if not os.path.exists(path2):
    os.makedirs(path2)

transform = transforms.Compose([transforms.CenterCrop(180),
                                transforms.Resize(256),
                                transforms.ToTensor()])

for item in os.listdir(path):
    img = Image.open(path + '/' + item)
    imgs = transform(img)
    save_image(imgs.data, '%s/%s' % (path2, item))
