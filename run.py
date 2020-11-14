import os
import argparse
from trainer import SemanticSeg
import pandas as pd
import random

from config import INIT_TRAINER, SETUP_TRAINER, VERSION, CURRENT_FOLD, LAB_LIST,IMG_LIST, FOLD_NUM

import time


def get_cross_validation(path_list, fold_num, current_fold):

    _len_ = len(path_list) // fold_num
    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(path_list[start_index:])
        train_id.extend(path_list[:start_index])
    else:
        validation_id.extend(path_list[start_index:end_index])
        train_id.extend(path_list[:start_index])
        train_id.extend(path_list[end_index:])

    print(len(train_id), len(validation_id))
    return train_id, validation_id


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train_cross_val',
                        choices=["train", 'train_cross_val', "inf"],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train_cross_val':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
    img_list = IMG_LIST
    lab_list = LAB_LIST
    # Training
    ###############################################
    if args.mode == 'train_cross_val':
        # path_list = path_list[:int(len(path_list) * 0.8)]
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_img_path, val_img_path = get_cross_validation(img_list, FOLD_NUM, current_fold)
            train_lab_path, val_lab_path = get_cross_validation(lab_list, FOLD_NUM, current_fold)
            SETUP_TRAINER['train_path'] = [train_img_path,train_lab_path]
            SETUP_TRAINER['val_path'] = [val_img_path,val_lab_path]
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        # path_list = path_list[:int(len(path_list) * 0.8)]
        train_img_path, val_img_path = get_cross_validation(img_list, FOLD_NUM,CURRENT_FOLD)
        train_lab_path, val_lab_path = get_cross_validation(lab_list, FOLD_NUM,CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = [train_img_path,train_lab_path]
        SETUP_TRAINER['val_path'] = [val_img_path,val_lab_path]
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    elif args.mode == 'inf':
        test_path ='./dataset/img_testA'
        print("test set len:",len(test_path))
        save_path = './result/testA'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        start_time = time.time()
        result = segnetwork.inference(test_path,save_path)
        print('run time:%.4f' % (time.time() - start_time))
        print('ave dice:%.4f' % (result))
    ###############################################
