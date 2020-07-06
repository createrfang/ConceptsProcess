# -*- coding: UTF-8 -*-

import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import my_model as models
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import training
from log import create_logger
import numpy as np
import shutil
import matplotlib.pyplot as plt
from helpers import *

# choose gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_dir = r'/home/zhuminqin/Code/ConceptsProcess/saved_models/001'
mkdir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'my_model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'training.py'), dst=model_dir)

# log
log, log_close = create_logger(log_filename=os.path.join(model_dir, 'test.log'))
# load data
test_dir = r'/home/zhuminqin/Data/dog_demo/dog_960'
test_batch_size = 16
test_datasets = datasets.ImageFolder(
    test_dir,
    transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    test_datasets, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)


# load model
net = models.vgg11_bn(pretrained=False, progress=True)
print(net)
pretrained_dict = torch.load('/home/zhuminqin/Code/ConceptsProcess/pretrained_models/vgg11_bn-6002323d.pth')
model_dict = net.state_dict()
print(len(pretrained_dict))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print(len(pretrained_dict))

net = net.cuda()
net_multi = nn.DataParallel(net)

# extract feature maps
net_multi.eval()
fts = training.train_or_test(net_multi, test_loader)
np.save(os.path.join(model_dir, 'fts.npy'), fts)

# fts clustering
# fts flatten
print(f'fts.shape :{fts.shape}')
fts = np.ndarray(fts).reshape((fts.shape[0], -1))
print(f'fts.shape :{fts.shape}')

# Dimensionality reduction
ts = TSNE(n_components=2)
new_ft = ts.fit_transform(fts)
print(type(new_ft))
print(f'new_ft.shape: {new_ft}')

# clustering
km = KMeans(n_clusters=10)
# np.save(dest_path, imgs)
km.fit(new_ft)
y_kmeans = km.predict(new_ft)
print(f'y_kmeans:{y_kmeans}')
plt.scatter(new_ft[:, 0], new_ft[:, 1], c=y_kmeans)
centers = km.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='k', s=200, alpha=0.5)
plt.savefig(os.path.join(model_dir, 'fig2.png'))
np.save(os.path.join(model_dir, 'new_ft.npy'), new_ft)
np.save(os.path.join(model_dir, 'y_kmeans.npy'), y_kmeans)
np.save(os.path.join(model_dir, 'centers.npy'), centers)

plt.scatter(new_ft[:, 0], new_ft[:, 1], s=10, alpha=0.8)
plt.scatter(centers[:, 0], centers[:, 1], c='k', s=100, alpha=0.1)
plt.imsave(os.path.join(model_dir, 'fig.png'))
# plt.show()


