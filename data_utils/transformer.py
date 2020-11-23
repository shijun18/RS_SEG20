import numpy as np
import torch
from PIL import Image
import random


class RandomFlip2D(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):

        image = sample['image']
        mask = sample['mask']

        if 'h' in self.mode and 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        elif 'h' in self.mode:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        elif 'v' in self.mode:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        new_sample = {'image': image, 'mask': mask}

        return new_sample


class RandomRotate2D(object):
    """
    Data augmentation method.
    Rotating the image with random degree.
    Args:
    - degree: the rotate degree from (-degree , degree)
    Returns:
    - rotated image and mask
    """

    def __init__(self, degree=[-15,-10,-5,0,5,10,15]):
        self.degree = degree

    def __call__(self, sample):
        image = sample['image']
        mask = sample['mask']

        rotate_degree = random.choice(self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        new_sample = {'image': image, 'mask': mask}
        return new_sample