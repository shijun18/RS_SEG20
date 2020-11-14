import os,glob
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list



def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = glob.glob(os.path.join(ckpt_path,'*.pth'))
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            return pth_list[-1]
        else:
            return None
    else:
        return None
    

def remove_weight_path(ckpt_path,retain=10):

    if os.path.isdir(ckpt_path):
        pth_list = glob.glob(os.path.join(ckpt_path,'*.pth'))
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split(':')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(pth_item)


def dfs_remove_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path)
        else:
            remove_weight_path(ckpt_path)
            break  

if __name__ == "__main__":

    ckpt_path = './ckpt/'
    dfs_remove_weight(ckpt_path)