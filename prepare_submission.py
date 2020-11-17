import os
from PIL import Image 
import numpy as np


input_path = './result/testA'
save_path = './result/submit_testA'

for item in os.scandir(input_path):
    img = Image.open(item.path)
    img_array = np.array(img)
    img_array[img_array == 255] = 0
    new_img = Image.fromarray(img_array,mode='P')
    new_img.save(os.path.join(save_path,item.name))
