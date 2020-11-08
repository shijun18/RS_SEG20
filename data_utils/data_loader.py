import os
import sys
sys.path.append('..')

from torch.utils.data import Dataset
import torch
import numpy as np

from PIL import Image



class To_Tensor(object):
  '''
  Convert the data in sample to torch Tensor.
  Args:
  - n_class: the number of class
  '''
  def __init__(self,num_class=2):
    self.num_class = num_class

  def __call__(self,sample):

    image = np.array(sample['image'],dtype=np.float32) / 255
    label = np.array(sample['label'],dtype=np.float32)
    
    # expand dims
    if len(image.shape) == 2:
        new_image = np.expand_dims(image,axis=0)
    else:
        new_image = image.transpose((2,0,1))
    new_label = np.empty((self.num_class,) + label.shape, dtype=np.float32)
    for z in range(1, self.num_class):
        temp = (label==z-1).astype(np.float32)
        new_label[z,...] = temp
    new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
   
    # convert to Tensor
    new_sample = {'image': torch.from_numpy(new_image),
                  'label': torch.from_numpy(new_label)}
    
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
  def __init__(self, img_list=None, lab_list=None, roi_number=None, num_class=2, transform=None):

    self.img_list = img_list
    self.lab_list = lab_list
    self.roi_number = roi_number
    self.num_class = num_class
    self.transform = transform


  def __len__(self):
    assert len(self.img_list) == len(self.lab_list),"The number of images and annotations should be the same."
    return len(self.img_list)


  def __getitem__(self,index):
    # Get image and label
    image = Image.open(self.img_list[index])
    label = Image.open(self.lab_list[index])
    assert os.path.splitext(os.path.basename(self.img_list[index]))[0] == os.path.splitext(os.path.basename(self.lab_list[index]))[0]
    if self.roi_number is not None:
        assert self.num_class == 2
        label = Image.fromarray((np.array(label)!=self.roi_number).astype(np.uint8)) 

    sample = {'image':image, 'label':label}
    # Transform
    if self.transform is not None:
      sample = self.transform(sample)

    return sample

