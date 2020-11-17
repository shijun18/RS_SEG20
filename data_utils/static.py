import sys
import os
import pandas as pd 
import numpy as np
from PIL import Image



def csv_maker(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        tag_array = np.zeros((len(label_list),),dtype=np.uint8)
        label = Image.open(item.path)
        label = np.array(label)
        label[label == 255] = 7
        tag_array[np.unique(label).astype(np.uint8)] = 1
        csv_item.extend(list(tag_array))

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)

def area_count(input_path,save_path,label_list):
    csv_info = []
    entry = os.scandir(input_path)
    len_ = len(os.listdir(input_path))
    count = 0 
    
    for item in entry:
        csv_item = []
        csv_item.append(item.path)
        area_array = np.zeros((len(label_list),),dtype=np.float32)
        label = Image.open(item.path)
        label = np.array(label)
        label[label == 255] = 7
        for i in range(len(label_list)):
            area_array[i] = np.sum((label==i).astype(np.float32))
        
        csv_item.extend(list(area_array))

        csv_info.append(csv_item)
        count += 1
        sys.stdout.write('\r Current: %d / %d'%(count,len_))
    print('\n')
    col = ['path'] + label_list
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv(save_path, index=False)


if __name__ == "__main__":
    lab_path = '/staff/shijun/torch_projects/RSI_SEG20/dataset/lab_train'
    csv_path = './area.csv'
    # csv_path = './annotation.csv'
    label_list = ['A','B','C','D','E','F','G','H']
    # csv_maker(lab_path,csv_path,label_list)
    area_count(lab_path,csv_path,label_list)
