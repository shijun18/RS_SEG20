import os
import sys
sys.path.append('..')

from torch.utils.data import Dataset
import torch
import numpy as np
import random

from PIL import Image


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''
    def __init__(self, num_class=2):
        self.num_class = num_class

    def __call__(self, sample):

        image = np.array(sample['image'], dtype=np.float32) / 255
        mask = np.array(sample['mask'], dtype=np.float32)

        # expand dims
        if len(image.shape) == 2:
            new_image = np.expand_dims(image, axis=0)
        else:
            new_image = image.transpose((2, 0, 1))
        new_mask = np.empty((self.num_class, ) + mask.shape, dtype=np.float32)
        for z in range(1,self.num_class):
            temp = (mask == z).astype(np.float32)
            new_mask[z, ...] = temp
        new_mask[0,...] = np.amax(new_mask[1:, ...],axis=0) == 0
        # convert to Tensor
        new_sample = {
            'image': torch.from_numpy(new_image),
            'mask': torch.from_numpy(new_mask)
        }

        return new_sample


class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - img_list: list of image path
    - lab_list: list of annotation path
    - roi_number: integer or None, to extract the corresponding label
    - num_class: the number of classes of the label
    - transform: the data augmentation methods
    '''
    def __init__(self,
                 img_list=None,
                 lab_list=None,
                 roi_number=None,
                 num_class=2,
                 input_channels=1,
                 transform=None,
                 crop_and_resize=False,
                 input_shape=None):

        self.img_list = img_list
        self.lab_list = lab_list
        self.roi_number = roi_number
        self.num_class = num_class
        self.input_channels = input_channels
        self.transform = transform
        self.crop_and_resize = crop_and_resize
        self.input_shape = input_shape

    def __len__(self):
        assert len(self.img_list) == len(self.lab_list), "The numbers of images and annotations should be the same."
        return len(self.img_list)

    def __getitem__(self, index):
        # Get image and mask
        if self.input_channels == 1:
            image = Image.open(self.img_list[index]).convert('L')
        else:
            image = Image.open(self.img_list[index])
        mask = Image.open(self.lab_list[index])
        # print(self.img_list[index])
        # print(self.lab_list[index])
        if self.crop_and_resize:
            image, mask = self._crop_and_resize(image,mask)
        assert os.path.splitext(os.path.basename(self.img_list[index]))[0] == os.path.splitext(os.path.basename(self.lab_list[index]))[0]
        if self.roi_number is not None:
            assert self.num_class == 2
            mask = Image.fromarray((np.array(mask) == self.roi_number).astype(np.uint8))
        else:
            mask_array = np.array(mask) + 1
            mask_array[mask_array > 255] = 0
            mask = Image.fromarray(mask_array.astype(np.uint8))

        sample = {'image': image, 'mask': mask}
        if self.transform is not None:
            sample = self.transform(sample)

        label = np.zeros((self.num_class, ), dtype=np.float32)
        label_array = np.argmax(sample['mask'].numpy(),axis=0)
        label[np.unique(label_array).astype(np.uint8)] = 1

        sample['label'] = torch.Tensor(list(label))
        # print(np.unique(np.argmax(np.array(sample['mask']),axis=0)))

        return sample
    
    def _crop_and_resize(self, image, mask, ratio=4):
        assert self.input_shape is not None
        height, weight = self.input_shape[0] // ratio, self.input_shape[1] // ratio
        left = random.choice([np.random.randint(0,(self.input_shape[1] - weight)//2),
                              np.random.randint((self.input_shape[1] + weight)//2,self.input_shape[1] - weight)])
        upper = random.choice([np.random.randint(0,(self.input_shape[0] - height)//2),
                               np.random.randint((self.input_shape[0] + height)//2,self.input_shape[0] - height)])
        image = image.crop((left,upper,left+weight,upper+height))
        mask = mask.crop((left,upper,left+weight,upper+height))
        
        # resize
        image = image.resize(self.input_shape, Image.BILINEAR)
        mask = mask.resize(self.input_shape, Image.NEAREST)
        return image, mask


