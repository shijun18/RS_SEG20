import sys
import os
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import binary_dilation
import time


def result_fusion(data_list,label_list=None,save_path=None):
    len_ = len(os.listdir(data_list[0]))
    count = 0
    for item in os.scandir(data_list[0]):
        img_list = [item.path] + [os.path.join(case, item.name) for case in data_list[1:]]
        palette = Image.open('./result/pspnet/results/A151678.png').getpalette()
        mask = np.array(Image.open(img_list[0]),dtype=np.uint8)
        for label in label_list:
            tmp_mask = np.zeros_like(mask,dtype=np.uint8)
            for img_path in img_list:
                tmp_mask += (np.array(Image.open(img_path)) == label).astype(np.uint8)
            binary_mask = (tmp_mask > len(data_list)/2).astype(np.uint8)
            if label == 4 or label==5:
                binary_mask = binary_dilation(binary_mask)
            mask[binary_mask == 1] = label
        mask = Image.fromarray(mask,mode='P')
        mask.putpalette(palette)
        mask.save(os.path.join(save_path,item.name))
        count += 1
        sys.stdout.write('\rCurrent %d/%d'%(count, len_))

    sys.stdout.write('\n')


def result_fusion_v2(data_list,label_list=None,save_path=None,shape=(256,256),weight=None):
    len_ = len(os.listdir(data_list[0]))
    count = 0
    for item in os.scandir(data_list[0]):
        img_list = [item.path] + [os.path.join(case, item.name) for case in data_list[1:]]
        palette = Image.open('./result/pspnet/results/A151678.png').getpalette()
        mask = np.zeros((len(label_list),) + shape,dtype=np.uint8)
        for i, img_path in enumerate(img_list):
            tmp_mask = np.zeros_like(mask,dtype=np.uint8)
            for label in label_list:
                temp = (np.array(Image.open(img_path)) == label).astype(np.uint8)
                tmp_mask[label,...] = temp
            mask[tmp_mask == 1] += weight[i]

        mask = Image.fromarray(np.argmax(mask,axis=0),mode='P')
        mask.putpalette(palette)
        mask.save(os.path.join(save_path,item.name))
        count += 1
        sys.stdout.write('\rCurrent %d/%d'%(count, len_))

    sys.stdout.write('\n')

if __name__ == "__main__":
    
    start = time.time()
    result_list = ['./result/t4/results','./result/t3/results','./result/pspnet/results','./result/deeplab_rs/results']
    # result_fusion(result_list,list(range(7)),'./result/results')
    result_fusion_v2(result_list,list(range(7)),'./result/results',weight=[9,8,7,5])
    print('Run time: %.4f'%(time.time() - start))