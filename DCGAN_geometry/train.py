import torch
import torch.nn.functional as F
from torch.optim import Adam
import scipy.sparse as sp
import numpy as np
import os
import time
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import args
from torch.autograd import Variable
import torchvision.transforms as transforms
from networks_ori import autoencoder


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if not os.path.exists(args.log_txt):
    os.mknod(args.log_txt)
if not os.path.exists(args.save_image_path):
    os.makedirs(args.save_image_path)
if not os.path.exists(args.save_model_path):
    os.makedirs(args.save_model_path)

train_loader = DataLoader(datasets.CIFAR10(args.dataroot, train=True,
                                           transform=transforms.Compose([
                                           transforms.Resize(args.image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
                                           download=False),batch_size=args.batch_size, shuffle=True, drop_last=True)

loss_fn = torch.nn.MSELoss()
loss_fn = loss_fn.cuda()

# init model and optimizer
model = autoencoder()
model = model.cuda()
optimizer = Adam(model.parameters(), lr=args.learning_rate)

# train model
f = open(args.log_txt, 'w')
for epoch in range(args.num_epoch):
    for i, (image, _) in enumerate(train_loader):
        t = time.time()
        real_image = Variable(image.type(torch.cuda.FloatTensor))
        optimizer.zero_grad()

        fake_image = model(real_image)
        loss = 0.5 * loss_fn(real_image, fake_image)
        loss.backward()
        optimizer.step()

        save_image(fake_image.data, '%s/%d_fake_image.png' % (args.save_image_path, epoch), nrow=100, normalize=True)
        torch.save(model.state_dict(), '%s/%d_fake_image.pkl' % (args.save_model_path, epoch))
        t2 = time.time()
        print("Epoch:", '%03d' % (epoch + 1), "Batch:", '%05d' % (i + 1), "train_loss=", "{:.5f}".format(loss.item()),
              "time=", "{:.5f}".format(t2 - t))
    f.writelines('[Epoch]: %04d [train_loss]: %.5f [time]: %.5f'
                 % (epoch + 1, loss.item(), (t2 - t)))

f.close()