import os
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import math
import shutil

from torch.nn import functional as F

from data_utils.transformer import RandomFlip2D, RandomRotate2D
from data_utils.data_loader import DataGenerator, To_Tensor

import torch.distributed as dist
# GPU version.


class SemanticSeg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    - mode: string __all__ = ['cls','seg','cls_and_seg','cls_or_seg']
    '''
    def __init__(self,
                 net_name=None,
                 lr=1e-3,
                 n_epoch=1,
                 channels=1,
                 num_classes=2,
                 roi_number=1,
                 input_shape=None,
                 batch_size=6,
                 num_workers=0,
                 device=None,
                 pre_trained=False,
                 weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 mode='cls'):
        super(SemanticSeg, self).__init__()

        self.net_name = net_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.channels = channels
        self.num_classes = num_classes
        self.roi_number = roi_number
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.net = self._get_net(self.net_name)
        self.pre_trained = pre_trained
        self.weight_path = weight_path

        self.start_epoch = 0
        self.global_step = 0
        self.loss_threshold = 1.0

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max

        self.mode = mode

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        if self.pre_trained:
            self._get_pre_trained(self.weight_path)
            # self.loss_threshold = eval(os.path.splitext(self.weight_path.split(':')[-1])[0])

        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

    def trainer(self,
                train_path,
                val_path,
                cur_fold,
                output_dir=None,
                log_dir=None,
                optimizer='Adam',
                loss_fun='Cross_Entropy',
                class_weight=None,
                lr_scheduler=None):

        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

        if not self.pre_trained:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
            os.makedirs(log_dir)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.writer = SummaryWriter(log_dir)
        self.global_step = self.start_epoch * math.ceil(
            len(train_path[0]) / self.batch_size)

        net = self.net
        lr = self.lr
        loss = self._get_loss(loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # dataloader setting
        if self.mode == 'cls':
            train_transformer = transforms.Compose([
                RandomRotate2D(),
                RandomFlip2D(mode='hv'),
                To_Tensor(num_class=self.num_classes)
            ])
        else:
            train_transformer = transforms.Compose([
                RandomFlip2D(mode='hv'),
                To_Tensor(num_class=self.num_classes)
            ])

        train_dataset = DataGenerator(train_path[0],
                                      train_path[1],
                                      roi_number=self.roi_number,
                                      num_class=self.num_classes,
                                      transform=train_transformer,
                                    #   crop_and_resize=True,
                                      input_shape=self.input_shape)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  pin_memory=True)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)
        if self.pre_trained:
            checkpoint = torch.load(self.weight_path)
            optimizer.load_state_dict(checkpoint['optimizer'])

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)

        # loss_threshold = 1.0
        for epoch in range(self.start_epoch, self.n_epoch):
            train_loss, train_dice, train_acc = self._train_on_epoch(epoch, net, loss, optimizer, train_loader)

            torch.cuda.empty_cache()

            val_loss, val_dice, val_acc = self._val_on_epoch(epoch, net, loss, val_path, train_transformer)

            torch.cuda.empty_cache()

            if lr_scheduler is not None:
                lr_scheduler.step(val_loss)

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))

            print('epoch:{},train_dice:{:.5f},val_dice:{:.5f}'.format(epoch, train_dice, val_dice))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('data/acc', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'],epoch)

            if val_loss <= self.loss_threshold:
                self.loss_threshold = val_loss

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': output_dir,
                    'state_dict': state_dict,
                    'optimizer': optimizer.state_dict()
                }

                file_name = 'epoch:{}-train_loss:{:.5f}-train_dice:{:.5f}-train_acc:{:.5f}-val_loss:{:.5f}-val_dice:{:.5f}-val_acc:{:.5f}.pth'.format(
                    epoch, train_loss, train_dice, train_acc, val_loss,
                    val_dice, val_acc)
                
                save_path = os.path.join(output_dir, file_name)
                print("save as %s" % file_name)

                torch.save(saver, save_path)

        self.writer.close()

    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader):

        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()
        train_acc = AverageMeter()

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['mask']
            label = sample['label']

            data = data.cuda()
            target = target.cuda()
            label = label.cuda()

            output = net(data)
            if self.mode == 'cls':
                loss = criterion(output[0], label)
            elif self.mode == 'seg':
                loss = criterion(output[1], target)
            else:
                loss = criterion(output,[label,target])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cls_output = output[0] #N*C
            cls_output = F.sigmoid(cls_output).float()

            seg_output = output[1].float() #N*C*H*W
            loss = loss.float()

            # measure acc
            acc = accuracy(cls_output.data, label)
            train_acc.update(acc.item(), data.size(0))

            # measure dice and record loss
            dice = compute_dice(seg_output.data, target, ignore_index=self.num_classes-1)
            # dice = compute_iou(seg_output.data, target, ignore_index=self.num_classes-1)
            train_loss.update(loss.item(), data.size(0))
            train_dice.update(dice.item(), data.size(0))

            torch.cuda.empty_cache()

            if self.mode == 'cls':
                print('epoch:{},step:{},train_loss:{:.5f},train_acc:{:.5f},lr:{}'.format(epoch, step, loss.item(), acc.item(), optimizer.param_groups[0]['lr']))

            elif self.mode == 'seg':
                print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},lr:{}'.format(epoch, step, loss.item(), dice.item(), optimizer.param_groups[0]['lr']))
            else:
                print('epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_acc:{:.5f},lr:{}'.format(epoch, step, loss.item(), dice.item(),acc.item(), optimizer.param_groups[0]['lr']))

            if self.global_step % 10 == 0:
                self.writer.add_scalars('data/train_loss_dice', {
                    'train_loss': loss.item(),
                    'train_dice': dice.item(),
                    'train_acc': acc.item()
                }, self.global_step)

            self.global_step += 1

        return train_loss.avg, train_dice.avg, train_acc.avg

    def _val_on_epoch(self, epoch, net, criterion, val_path, val_transformer):

        net.eval()

        val_dataset = DataGenerator(val_path[0],
                                    val_path[1],
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=val_transformer,
                                    # crop_and_resize=True,
                                    input_shape=self.input_shape)

        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        val_loss = AverageMeter()
        val_dice = AverageMeter()
        val_acc = AverageMeter()

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['mask']
                label = sample['label']

                data = data.cuda()
                target = target.cuda()
                label = label.cuda()

                output = net(data)
                if self.mode == 'cls':
                    loss = criterion(output[0], label)
                elif self.mode == 'seg':
                    loss = criterion(output[1], target)
                else:
                    loss = criterion(output,[label,target])


                cls_output = output[0]
                cls_output = F.sigmoid(cls_output).float()

                seg_output = output[1].float()
                loss = loss.float()

                # measure acc
                acc = accuracy(cls_output.data, label)
                val_acc.update(acc.item(),data.size(0))

                # measure dice and record loss
                dice = compute_dice(seg_output.data, target, ignore_index=self.num_classes-1)
                # dice = compute_iou(seg_output.data, target, ignore_index=self.num_classes-1)
                val_loss.update(loss.item(), data.size(0))
                val_dice.update(dice.item(), data.size(0))

                torch.cuda.empty_cache()

                if self.mode == 'cls':
                    print('epoch:{},step:{},val_loss:{:.5f},val_acc:{:.5f}'.format(epoch, step, loss.item(), acc.item()))
                elif self.mode == 'seg':
                    print('epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f}'.format(epoch, step, loss.item(), dice.item()))
                else:
                    print('epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},val_acc:{:.5f}'.format(epoch, step, loss.item(), dice.item(), acc.item()))

        return val_loss.avg, val_dice.avg, val_acc.avg

    def test(self, test_path, net=None, mode='seg'):
        if net is None:
            net = self.net
        
        net = net.cuda()
        net.eval()
        
        if self.mode == 'cls':
            test_transformer = transforms.Compose([
                RandomRotate2D(),
                RandomFlip2D(mode='hv'),
                To_Tensor(num_class=self.num_classes)
            ])
        else:
            test_transformer = transforms.Compose([
                RandomFlip2D(mode='hv'),
                To_Tensor(num_class=self.num_classes)
            ])

        test_dataset = DataGenerator(test_path[0],
                                    test_path[1],
                                    roi_number=self.roi_number,
                                    num_class=self.num_classes,
                                    transform=test_transformer)

        test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=self.num_workers,
                                pin_memory=True)

        test_iou = AverageMeter()
        test_dice = AverageMeter()
        test_acc = AverageMeter()
        from metrics import RunningConfusionMatrix
        cm = RunningConfusionMatrix(labels=list(range(7)),ignore_label=self.num_classes-1)

        with torch.no_grad():
            for step, sample in enumerate(test_loader):
                data = sample['image']
                target = sample['mask']
                label = sample['label']
                print(label)

                data = data.cuda()
                target = target.cuda()
                label = label.cuda()

                output = net(data)

                cls_output = output[0]
                cls_output = F.sigmoid(cls_output).float()

                seg_output = output[1].float()

                # measure acc
                acc = accuracy(cls_output.data, label)
                test_acc.update(acc.item(),data.size(0))

                # measure dice and iou for evaluation (float)
                dice = compute_dice(seg_output.data, target, ignore_index=self.num_classes-1)
                iou = compute_iou(seg_output.data, target, ignore_index=self.num_classes-1)
                test_iou.update(iou.item(), data.size(0))
                test_dice.update(dice.item(), data.size(0))
                
                # mIoU (int)
                seg_output = F.softmax(seg_output, dim=1)
                cls_output = (cls_output > 0.5).float() # N*C
                print(cls_output.detach())
                if mode == 'mtl':
                    b, c, _, _ = seg_output.size()
                    seg_output = seg_output * cls_output.view(b,c,1,1).expand_as(seg_output)

                seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()  #N*H*W N=1
                target = torch.argmax(target,1).detach().cpu().numpy()

                print(np.unique(seg_output),np.unique(target))
                
                cm.update_matrix(target.flatten(),seg_output.flatten())
                

                torch.cuda.empty_cache()
                
                print('step:{},test_iou:{:.5f},test_dice:{:.5f},test_acc:{:.5f}'.format(step,iou.item(), dice.item(),acc.item()))
            
            miou = cm.compute_current_mean_intersection_over_union()
            print('miou:{:.5f},avg_iou:{:.5f},avg_dice:{:.5f},avg_acc:{:.5f}'.format(miou,test_iou.avg, test_dice.avg, test_acc.avg))

    def inference(self, test_path, save_path, net=None, palette=None):

        if net is None:
            net = self.net

        net = net.cuda()
        net.eval()

        from PIL import Image

        with torch.no_grad():
            for item in os.scandir(test_path):
                img = Image.open(item.path)
                assert len(np.array(img).shape) == 3
                # normalization and transpose
                data_array = (np.array(img, dtype=np.float32) / 255).transpose((2, 0, 1))  #C*H*W
                data_array = np.expand_dims(data_array, axis=0)  #N*C*H*W N=1
                data = torch.from_numpy(data_array)

                data = data.cuda()
                output = net(data)

                seg_output = output[1]  #N*C*H*W N=1
                seg_output = F.softmax(seg_output, dim=1)

                cls_output = output[0] # N*C
                cls_output = F.sigmoid(cls_output) # N*C
                cls_output = (cls_output > 0.5).float() # N*C


                if self.mode == 'mtl':
                    b, c, _, _ = seg_output.size()
                    seg_output = seg_output * cls_output.view(b,c,1,1).expand_as(seg_output)
                    cls_output = cls_output.squeeze().detach().tolist()
                    print("cls result:",cls_output)

                seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()  #N*H*W N=1
                print(np.unique(seg_output))
                seg_output = np.squeeze(seg_output).astype(np.uint8)  #H*W
                seg_output[seg_output > 6] = 255
                seg_output = Image.fromarray(seg_output, mode='P')
                if palette is not None:
                    seg_output.putpalette(palette)

                # save
                seg_output.save(os.path.join(save_path,item.name.split('.')[0] + '.png'))
                torch.cuda.empty_cache()

    def _get_net(self, net_name):
        if net_name == 'c_unet':
            from model.unet import unet
            net = unet(n_channels=self.channels, n_classes=self.num_classes)
        elif net_name == 'r_unet':
            from model.unet import unet
            net = unet(n_channels=self.channels, n_classes=self.num_classes, revise=True)
        elif net_name == 'e_unet':
            from model.unet import unet
            net = unet(n_channels=self.channels, n_classes=self.num_classes, cls_location='end')
        return net

    def _get_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight, ignore_index=self.num_classes-1)
        elif loss_fun == 'TopKLoss':
            from loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight, ignore_index=self.num_classes-1, k=20)
        elif loss_fun == 'DiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=self.num_classes-1, p=1)
        
        elif loss_fun == 'mIoU_loss':
            from loss.mIoU_loss import mIoU_loss
            loss = mIoU_loss(weight=class_weight, ignore_index=self.num_classes-1, p=1)

        elif loss_fun == 'PowDiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=self.num_classes-1, p=2)

        elif loss_fun == 'BCEWithLogitsLoss':
            loss = nn.BCEWithLogitsLoss(class_weight)
        
        elif loss_fun == 'BCEPlusDice':
            from loss.combine_loss import BCEPlusDice
            loss = BCEPlusDice(weight=class_weight,ignore_index=self.num_classes-1,p=2)

        return loss

    def _get_optimizer(self, optimizer, net, lr):
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(),
                                        lr=lr,
                                        momentum=self.momentum)

        return optimizer

    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.T_max)
        return lr_scheduler

    def _get_pre_trained(self, weight_path):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1


# computing tools


class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def binary_iou(predict, target, smooth=1e-5):
    """IoU of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        IoU tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    inter = torch.sum(torch.mul(predict, target), dim=1)
    union = torch.sum(predict + target, dim=1)

    iou = (inter + smooth) / (union - inter + smooth)

    return iou.mean()



def binary_dice(predict, target, smooth=1e-5):
    """Dice of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        dice according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    inter = torch.sum(torch.mul(predict, target), dim=1)
    union = torch.sum(predict + target, dim=1)

    dice = (2 * inter + smooth) / (union + smooth)

    return dice.mean()


def compute_dice(predict, target, ignore_index=0):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    total_dice = 0.
    predict = F.softmax(predict, dim=1)
    # predict = F.sigmoid(predict)
    dice_list = []
    for i in range(target.shape[1]):
        if i != ignore_index:
            dice = binary_dice(predict[:, i], target[:, i])
            # print(dice)
            total_dice += dice
            dice_list.append(round(dice.item(),4))
    print(dice_list)

    if ignore_index is not None:
        return total_dice / (target.shape[1] - 1)
    else:
        return total_dice / target.shape[1]


def compute_iou(predict, target, ignore_index=0):
    """
    Compute iou
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean iou over the batch
    """
    assert predict.shape == target.shape, 'predict & target shape do not match'
    total_iou = 0.
    predict = F.softmax(predict, dim=1)
    for i in range(target.shape[1]):
        if i != ignore_index:
            iou = binary_iou(predict[:, i], target[:, i])
            total_iou += iou

    if ignore_index is not None:
        return total_iou / (target.shape[1] - 1)
    else:
        return total_iou / target.shape[1]


def accuracy(output, target):
    '''
    Computes the precision@k for the specified values of k
    - output shape: N*C
    - target shape: N*C
    '''
    batch_size, class_num = target.size()
    pred = (output > 0.5).float()
    correct = pred.eq(target)
    acc = correct.float().sum() / (batch_size*class_num) 

    return acc
