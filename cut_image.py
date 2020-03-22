import torch
import os
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets

dir_name1 = "/media/ouc/4T_A/jiang/baseline/instagan/datasets/shp2gir_coco/trainA"
dir_name2 = "/media/ouc/4T_A/jiang/baseline/instagan/datasets/shp2gir_coco/trainA_seg"
dir_name3 = "/Users/jiangyufeng/Desktop/pants2skirt_mhp/newTrainA"

if not os.path.exists(dir_name3):
	os.makedirs(dir_name3)

dir1 = os.listdir(dir_name1)
dir2 = os.listdir(dir_name2)

print(dir1)
print(dir2)

transform = transforms.ToTensor()
# transform = transforms.Compose([#transforms.CenterCrop(256),
# 								transforms.ToTensor(),
# 								transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
for i in range(len(dir1)):
	# print(dir1[i])
	img1 = Image.open(dir_name1 + "/" + dir1[i]).convert('RGB')
	img2 = Image.open(dir_name2 + "/" + dir2[i]).convert('L')
	img1 = transform(img1)
	img2 = transform(img2) # 黑的地方tensor为0，白的地方tensor值为1
	# img1 = img1.numpy()
	# img2 = img2.numpy()
	# print(img2.shape)
	# img2 = img2.squeeze(0)
	for a in range(img2.size(1)):
		for b in range(img2.size(2)):
			if img2[0][a][b] != 0. and img2[0][a][b] != 1.:
				img2[0][a][b] = 1.
			# else:
			# 	img2[0][a][b] = 1.
	# img2 = img2.unsqueeze(0)
	# img2 = img2.numpy()
	# save_image(img2.data, '%s/%s' % (dir_name3, dir1[i]))

	# save_image(img1.data, '%s/%s' % (dir_name3, dir1[i]))
	img_new = img1 * img2
	for a in range(img_new.size(0)):
		for b in range(img_new.size(1)):
			for c in range(img_new.size(2)):
				if img_new[a][b][c] == 0.:
					img_new[a][b][c] = 1.

	# img_new = img_new.numpy()
	save_image(img_new.data, '%s/%s' % (dir_name3, dir1[i]))
	# print(img2.shape)
	# img1 = img1.numpy()
	# img2 = img2.numpy()
	# print(img2)
	# print(img1)