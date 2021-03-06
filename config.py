import os
import glob

from utils import get_path_with_annotation
from utils import get_weight_path


__net__ = ['c_unet','r_unet','e_unet']
__plan__ = ['all','single','seg_single']
__mode__ = ['cls','seg','mtl']


ROI_LIST = ['A','B','C','D','E','F','G']
    

PLAN = 'all'
NET_NAME = 'e_unet'
VERSION = 'v3.3'

# for the all and single plan, mode can be one of ['cls','seg','mtl'], 
# but when plan == seg_single, the mode must be 'seg'
MODE = 'seg'


DEVICE = '1'
# Must be True when pre-training and inference
PRE_TRAINED = True 
# 1,2,...,8
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 9

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None# or 0,1,2,3,4,5,6 
NUM_CLASSES = len(ROI_LIST) + 1 # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    NUM_CLASSES = 2
    ROI_NAME = ROI_LIST[ROI_NUMBER]
else:
    ROI_NAME = 'All'
#---------------------------------

#--------------------------------- mode and data path setting
if PLAN == 'seg_single':
    assert ROI_NUMBER is not None, "roi number must not be None in 2d clean"
    LAB_LIST = get_path_with_annotation('./data_utils/annotation.csv','path',ROI_NAME)
    LAB_LIST.sort()
    IMG_LIST = [os.path.join('/staff/shijun/torch_projects/RSI_SEG20/dataset/img_train',os.path.basename(case).split('.')[0] + '.jpg') for case in LAB_LIST]
else:
    IMG_LIST = glob.glob(os.path.join('./dataset/img_train','*.jpg'))
    IMG_LIST.sort()
    LAB_LIST = glob.glob(os.path.join('./dataset/lab_train','*.png'))
    LAB_LIST.sort()
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (256,256)
BATCH_SIZE = 32

# CKPT_PATH = '/staff/shijun/torch_projects/RSI_SEG20/ckpt/{}/{}/{}/{}/fold{}'.format(PLAN,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))
CKPT_PATH = '/staff/shijun/torch_projects/RSI_SEG20/ckpt/{}/{}/{}/{}/fold{}'.format(PLAN,'cls','v3.0',ROI_NAME,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'lr':1e-1, 
  'n_epoch':80,
  'channels':3,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER,
  'input_shape':INPUT_SHAPE,
  'batch_size':BATCH_SIZE,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.001,
  'momentum': 0.99,
  'gamma': 0.1,
  'milestones': [40,80],
  'T_max':5,
  'mode':MODE
 }
#---------------------------------

__seg_loss__ = ['DiceLoss','mIoU_loss','PowDiceLoss','Cross_Entropy','TopkDiceLoss','TopKLoss','CEPlusDice','TopkCEPlusDice','CEPlusTopkDice','TopkCEPlusTopkDice']
__cls_loss__ = ['BCEWithLogitsLoss']
__mtl_loss__ = ['BCEPlusDice']
# Arguments when perform the trainer 

if MODE == 'cls':
    LOSS_FUN = 'BCEWithLogitsLoss'
elif MODE == 'seg' :
    LOSS_FUN = 'Cross_Entropy'
else:
    LOSS_FUN = 'BCEPlusDice'

SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}/{}/{}'.format(PLAN,MODE,VERSION,ROI_NAME),
  'log_dir':'./log/{}/{}/{}/{}'.format(PLAN,MODE,VERSION,ROI_NAME), 
  'optimizer':'SGD',
  'loss_fun':LOSS_FUN,
  'class_weight':None, #[2,1,1,1,4,4,3,1]
  'lr_scheduler':'CosineAnnealingLR', #'CosineAnnealingLR'
  }
#---------------------------------
