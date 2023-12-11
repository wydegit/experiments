import numpy as np
import torch.cuda


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc




def parse_args():
    """
    Training option and segmentation experiments
    :return:
    """
    parser = argparse.ArgumentParser(description='alcnet pytorch')

    ####### model #######
    parser.add_argument('--net-choice', type=str, default='ALCNet', help='model')
    parser.add_argument('--pyramid-mode', type=str, default='Dec', help='Inc,Dec') # ?
    parser.add_argument('--r', type=int, default=2, help='choice:1,2,4')   # ?
    parser.add_argument('--summary', action='store_true', default=False, help='print parameters')   # 命令行输入参数则为True(激活)，否则为False
    parser.add_argument('--scale-mode', type=str, default='Multiple', help='choice:Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal', help='choice:add, max, sk')
    parser.add_argument('--cue', type=str, default='lcm', help='choice:lcm, orig')  # ?

    ####### dataset #######
    parser.add_argument('--data_root', type=str, default='./data/', help='dataset path')
    parser.add_argument('--out', type=str, default='./', help='metrics saved path')
    parser.add_argument('--dataset', type=str, default='open-sirst-v2', help='choice:DENTIST, Iceberg')
    parser.add_argument('--workers', type=int, default=1, metavar='N', help='dataloader threads')   # metavar ?
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--blocks', type=int, default=4, help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16, help='channels')
    parser.add_argument('--shift', type=int, default=13, help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--train-split', type=str, default='train_v1', help='choice:train, trainval')
    parser.add_argument('--val-split', type=str, default='val_v1', help='choice:test, val')


    ####### training hyperparameters #######
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,200', help='epochs at which learning rate decays (default: 40,60)')
    parser.add_argument('--gamma', type=int, default=2, help='gamma for Focal Soft Iou Loss')
    parser.add_argument('--lambda', type=int, default=1, help='lambda for TV Soft Iou Loss')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay')
    parser.add_argument('--no-wd', action='store_true', default=False, help='whether to remove weight decay on bias and beta/gamma for bn layers')
    parser.add_argument('--sparsity', action='store_true', help='whether to use sparsity regularization')   # ？
    parser.add_argument('--score-thresh', type=float, default=0.5, help='score-thresh')

    ####### cuda and logging #######
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--gpus', type=str, default='0', help='Training with which gpus like 0,1,2,3')
    parser.add_argument('--kvstore', type=str, default='device', help='kvstore to use for trainer/module.')  # multi-GPU training
    parser.add_argument('--dtype', type=str, default='float32', help='data type for training')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay rate')  # ? 与上边weight-decay有什么区别
    parser.add_argument('--log-dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--log-iter', type=int, default=10, help='print log every log-iter')



    ####### checking point #######
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')  # './params/
    parser.add_argument('--save-dir', type=str, default='./params', help='Directory for saving checkpoint models')
    parser.add_argument('--colab', action='store_true', help='whether using colab')

    ####### evaluation #######
    parser.add_argument('--eval', action='store_true', default=False, help='evaluating only')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')
    parser.add_argument('--metric', type=str, default='mAP', help='choich:F1, IoU, mAP')

    ####### synchronized BatchNorm for multiple devices or distributed system #######
    parser.add_argument('--syncbn', action='store_true', help='using Synchronized Cross-GPU BatchNorm')

    args = parser.parse_args()


    ## used devices  (ctx)
    # available_gpus = list(range(torch.cuda.device_count()))
    if args.no_cuda or (torch.cuda.is_available() == False):
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = torch.device('cpu')
    else:
        args.ctx = [torch.device('cuda:' + i) for i in args.gpus.split(',') if i.strip()]
        print('Using {} GPU: {}, '.format(len(args.ctx), args.ctx))

    ## Synchronized BatchNorm setting
    args.norm_layer = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}

    print(args)
    return args



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




class Visualize():
