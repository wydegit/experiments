
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchinfo import summary
from torchvision import transforms

from torch.utils.data import DataLoader

from mydataset import SIRST

from loss import SoftIoULoss
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import argparse


def parse_args():
    """
    Training option and segmentation experiments
    :return:
    """
    parser = argparse.ArgumentParser(description='alcnet pytorch')

    ####### model #######
    parser.add_argument('--net_choice', type=str, default='PCMNet', help='model')
    parser.add_argument('--pyramid-mode', type=str, default='Dec', help='Inc,Dec') # ?
    parser.add_argument('--r', type=int, default=2, help='choice:1,2,4')   # ?
    parser.add_argument('--summary', action='store_true', help='print parameters')   # 命令行输入参数则为True(激活)，否则为False
    parser.add_argument('--scale-mode', type=str, default='Multiple', help='choice:Single, Multiple, Selective')
    parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal', help='choice:add, max, sk')
    parser.add_argument('--cue', type=str, default='lcm', help='choice:lcm, orig')  # ?

    ####### dataset #######
    parser.add_argument('--data_root', type=str, default='./data/', help='dataset path')
    parser.add_argument('--dataset', type=str, default='SIRST', help='choice:DENTIST, Iceberg')
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')   # metavar ?
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--blocks', type=int, default=4, help='[1] * blocks')
    parser.add_argument('--channels', type=int, default=16, help='channels')
    parser.add_argument('--shift', type=int, default=13, help='shift')
    parser.add_argument('--iou-thresh', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--train-split', type=str, default='trainval', help='choice:train, trainval')
    parser.add_argument('--val-split', type=str, default='test', help='choice:test, val')

    ####### training hyperparameters #######
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,200', help='epochs at which learning rate decays (default: 40,60)')
    parser.add_argument('--gamma', type=int, default=2, help='gamma for Focal Soft Iou Loss')
    parser.add_argument('--lambda', type=int, default=1, help='lambda for TV Soft Iou Loss')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M', help='weight-decay')
    parser.add_argument('--no-wd', action='store_true', help='whether to remove weight decay on bias and beta/gamma for bn layers')
    parser.add_argument('--sparsity', action='store_true', help='whether to use sparsity regularization')   # ？
    parser.add_argument('--score-thresh', type=float, default=0.5, help='score-thresh')

    ####### cuda and logging #######
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--gpus', type=str, default='0', help='Training with which gpus like 0,1,2,3')
    parser.add_argument('--kvstore', type=str, default='device', help='kvstore to use for trainer/module.')  # ?
    parser.add_argument('--dtype', type=str, default='float32', help='data type for training')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay rate')  # ? 与上边weight-decay有什么区别

    ####### checking point #######
    parser.add_argument('--resume', type=str, default=None,help='put the path to resuming file if needed')
    parser.add_argument('--colab', action='store_true', help='whether using colab')

    ####### evaluation #######
    parser.add_argument('--eval', action='store_true', help='evaluating only')
    parser.add_argument('--no-val', action='store_true', help='skip validation during training')
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


class Trainer(object):
    def __init__(self, args):
        input_transform = transforms.Compose(
            [transforms.ToTensor(),
             # [.418, .447, .571], [.091, .078, .076] iceberg mean and std
             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        ######## dataset and dataloader ########
        data_root = args.data_root
        if os.path.exists(data_root) is False:
            raise FileNotFoundError("{} is not found".format(data_root))

        data_kwargs = {'root': data_root,
                       'base_dir': args.dataset,
                       'base_size': args.base_size,
                       'crop_size': args.crop_size,
                       'transform': input_transform,
                       'include_name': False}


        trainset = SIRST(split=args.train_split, mode='train', **data_kwargs)
        valset = SIRST(split=args.val_split, mode='testval', **data_kwargs)

        self.train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        self.train_data = DataLoader(trainset, args.batch_size, shuffle=True,
                                     last_batch='rollover', num_workers=args.workers)

        self.eval_data = DataLoader(valset, args.batch_size,
                                     last_batch='rollover', num_workers=args.workers)

        ######## model ########
        # net_choice = 'PCMNet'   # ResnetFPN, PCMNet, MPCMNet, LayerwiseMPCMNet
        net_choice = args.net_choice
        print("net_choice", net_choice)

        if net_choice == 'MPCMResnetFPN':
            r = args.r
            layers = [args.blocks] * 3    # 3 stage, each stage has args.blocks(4) resnet basic blocks
            channels = [8, 16, 32, 64]
            shift = args.shift
            pyramid_mode = args.pyramid_mode
            scale_mode = args.scale_mode
            pyramid_fuse = args.pyramid_fuse

            model = MPCMResnetFPN(layers=layers, channels=channels, shift=shift,
                                  pyramid_mode=pyramid_mode, scale_mode=scale_mode, pyramid_fuse=pyramid_fuse,
                                  r=r, classes=trainset.NUM_CLASSES)

            print("net_choice:{}\nscale_mode:{}\npyramid_fuse:{}\nr:{}\nlayers:{}\nchannels:{}\nshift:{}\n"
                  .format(net_choice, scale_mode, pyramid_fuse, r, layers, channels, shift))

        # self.host_name = socket.gethostname()
        self.save_prefix = self.host_name + '_' + net_choice + '_scale-mode_' + args.scale_mode + \
                           '_pyramid-fuse_' + args.pyramid_fuse + '_b_' + str(args.blocks)

        # if args.net_choice == 'ResNetFCN':
        #    self.save_prefix = self.host_name + '_' + net_choice + '_b_' + str(args.blocks)

        # resume checkpoint if needed 加载权重否则初始化
        if args.resume is not None:
            if os.path.isfile(args.resume):

                print("=> loading checkpoint '{}'".format(args.resume))
                model.load_state_dict(torch.load(args.resume), ctx=args.ctx)  # checkpoint->state_dict->weights  args.ctx?
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        else:
            model.apply(init_weights)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)

        self.net = model

        # summary
        if args.summary:
            summary(self.net, input_size=(args.batch_size, 3, args.crop_size, args.crop_size))
            # self.net.summary(mx.nd.ones((1, 3, args.crop_size, args.crop_size), ctx=args.ctx)) # args.ctx?

        # loss
        # self.criterion = SoftIoULoss()
        #
        # # lr_scheduler and optimizer
        # lr_lambda = lambda epoch: 1 - (epoch / args.epochs) ** 0.9
        # self.lr_scheduler =
        #
        # optimizer_params = {'wd': args.weight_decay,
        #                     'learning_rate': args.lr,}
        # if args.dtype == 'float16':
        #     optimizer_params['multi_precision'] = True
        # if args.no_wd:
        #     for k, v in self.net.params('.*beta|.*gamma|.*bias').items():
        #         v.wd_mult = 0.0  # ?
        #
        # self.optimizer = optim.Adagrad(self.net.parameters(), lr=optimizer_params['learning_rate'],
        #                           weight_decay=optimizer_params['wd'])

        ######### evaluation metrics #########


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.PReLU):
        init.constant_(m.weight, 0.25)



if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        trainer.validation(0)