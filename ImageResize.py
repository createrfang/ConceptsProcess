import torchvision.transforms as trans
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from helpers import *

def img_resize_save(files, size):
    dest_dir = os.path.join('/home/zhuminqin/Data/dog_demo/', 'dest_img'+str(size))
    mkdir(dest_dir)
    for file in files:
        if not os.path.isdir(os.path.join(img_dir, file)):
            f = Image.open(os.path.join(img_dir, file))
            f = f.resize((size, size))
            f.save(os.path.join(dest_dir, file + '.jpg'))


if __name__ == '__main__':
    img_dir = '/home/zhuminqin/Data/dog_demo/dog'
    imgs = os.listdir(img_dir)
    now_size = 960
    for i in range(6):
        print(now_size)
        img_resize_save(imgs, now_size)
        now_size = now_size // 2



