import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, PolynomialLR, CosineAnnealingLR
from module import Backbone
from model import ALCNet_straight, ALCNet_cross, ACMFPN_straight, ACMFPN_cross
from mydataset import SIRST
from loss import SoftIoULoss
from metric import *
from logger import setup_logger
from tqdm import tqdm
import shutil
from datetime import datetime
from visualize import train_visualize, val_visualize, plot_img_and_mask




def parse_args():
    """
    Training option and segmentation experiments
    :return: args
    """
    parser = argparse.ArgumentParser(description='alcnet pytorch')

    ######## model ########
    parser.add_argument('--net-choice', type=str, default='ALCNet-straight', help='model')
    parser.add_argument('--summary', action='store_true', default=True, help='print model summary')

    parser.add_argument('--blocks', type=int, default=4, help='[1] * ResnetBasicBlocks')
    parser.add_argument('--channels', type=int, default=8, help='stem channels')
    # parser.add_argument('--shift', type=int, default=13, help='lcm shift')
    # parser.add_argument('--r', type=int, default=2, help='choice:1,2,4')
    # parser.add_argument('--pyramid-mode', type=str, default='Dec', help='Inc,Dec')
    # parser.add_argument('--scale-mode', type=str, default='Multiple', help='choice:Single, Multiple, Selective')
    # parser.add_argument('--pyramid-fuse', type=str, default='bottomuplocal', help='choice:add, max, sk')
    parser.add_argument('--score-thresh', type=float, default=0.5, help='score-thresh')

    ######## dataset ########
    parser.add_argument('--data-root', type=str, default='./data/', help='dataset path')   # /content/experiments/data
    parser.add_argument('--dataset', type=str, default='open-sirst-v2', help='dataset name')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')  # 512
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')  # 480
    parser.add_argument('--workers', type=int, default=1, metavar='N', help='dataloader threads')
    parser.add_argument('--train-split', type=str, default='trainval_v1', help='choice:train, trainval')
    parser.add_argument('--val-split', type=str, default='test_v1', help='choice:test, val')
    parser.add_argument('--test-split', type=str, default='test_v1', help='choice:test, val')

    ######## training hyperparameters ########
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 1e-3)')   # 0.1
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR', help='min learning rate (default: 1e-6)')
    # parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='weight decay for loss regularization')
    parser.add_argument('--dtype', type=str, default='float32', help='data type for training')
    parser.add_argument('--no-wd', action='store_true', default=False, help='whether to remove weight decay on bias and beta/gamma for bn layers')

    ######## evaluation hyperparameters ########
    parser.add_argument('--eval', action='store_true', default=False, help='evaluating only')
    parser.add_argument('--no-val', action='store_true', default=False, help='skip validation during training')

    ######## testing hyperparameters ########
    parser.add_argument('--predict', default=False, action='store_true', help='test model')
    parser.add_argument('--predict-checkpoint', type=str, default='./params/ALCNet_open-sirst-v2_265.pth',
                        help='.pth used in model predict')

    ######## logging and checkpoint ########
    parser.add_argument('--log-dir', type=str, default='./logs', help='log directory')
    parser.add_argument('--save-dir', type=str, default='./params', help='Directory for saving checkpoint models')
    parser.add_argument('--visual-dir', type=str, default='./visual', help='Directory for saving visualization images')
    parser.add_argument('--visual-img', default=False, action='store_true', help='whether to visualize images')
    parser.add_argument('--colab', action='store_true', default=False, help='whether using colab')
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')

    ######## devices and distributed training ########
    parser.add_argument('--kvstore', type=str, default='device', help='kvstore to use for trainer/module.')
    parser.add_argument('--gpus', type=str, default='0', help='Training with which gpus like 0,1,2,3')

    parser.add_argument('--syncbn', action='store_true', help='using Synchronized Cross-GPU BatchNorm')


    args = parser.parse_args()
    if torch.cuda.is_available():
        args.ctx = [torch.device('cuda:' + i) for i in args.gpus.split(',') if i.strip()]
        print('Using {} GPU: {}, '.format(len(args.ctx), args.ctx))
    else:
        args.kvstore = 'local'
        args.ctx = torch.device('cpu')
        print('Using CPU')

    ######## Synchronized BatchNorm setting  ########
    args.norm_layer = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d
    args.norm_kwargs = {'num_devices': len(args.ctx)} if args.syncbn else {}

    print(args)
    return args


def save_checkpoint(model, args, epoch, is_best=False):
    """Save Checkpoint"""

    directory = args.save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.net_choice, args.dataset, epoch)
    filename = os.path.join(directory, filename)

    # if args.distributed:
    #     #     model = model.module
    torch.save(model.state_dict(), filename)

    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)




class Trainer(object):
    def __init__(self, args):

        ######## dataset and dataloader ########
        input_transform = transforms.Compose(
            [transforms.ToTensor(),
             # transforms.Normalize([.439], [.108])])   # sirst_trainvaltest_v1_single_channel mean and std
             # [.418, .447, .571], [.091, .078, .076] iceberg mean and std
             transforms.Normalize([.485, .456, .406], [.229, .224, .225])]) #imagenet mean and std

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
        valset = SIRST(split=args.val_split, mode='val', **data_kwargs)
        testset = SIRST(split=args.test_split, mode='testval', **data_kwargs)

        # args.iters_per_epoch = len(trainset) // (len(args.ctx) * args.batch_size)
        # args.max_iters = args.epochs * args.iters_per_epoch

        self.train_data = DataLoader(trainset, args.batch_size, shuffle=True,
                                     drop_last=True, num_workers=args.workers, pin_memory=True)

        self.eval_data = DataLoader(valset, args.batch_size, shuffle=False,
                                     drop_last=False, num_workers=args.workers, pin_memory=True)

        self.test_data = testset

        ######## model ########
        net_choice = args.net_choice
        print("net_choice:", net_choice)

        layers = [args.blocks] * 3   #
        channels = [8, 16, 32, 64]

        backbone = Backbone(layers, channels, tiny=False)

        if net_choice == 'ALCNet-cross':
            self.net = ALCNet_cross(backbone=backbone)
        elif net_choice == 'ALCNet-straight':
            self.net = ALCNet_straight(backbone=backbone)
        elif net_choice == 'ACMFPN-cross':
            self.net = ACMFPN_cross(backbone=backbone)
        elif net_choice == 'ACMFPN-straight':
            self.net = ACMFPN_straight(backbone=backbone)
        else:
            raise ValueError('Unknown net_choice: {}'.format(net_choice))


        # self.host_name = socket.gethostname()
        # self.save_prefix = self.host_name + '_' + net_choice + '_scale-mode_' + args.scale_mode + \
        #                    '_pyramid-fuse_' + args.pyramid_fuse + '_b_' + str(args.blocks)

        # if args.net_choice == 'ResNetFCN':
        #    self.save_prefix = self.host_name + '_' + net_choice + '_b_' + str(args.blocks)


        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                self.net.load_state_dict(torch.load(args.resume))  # checkpoint->state_dict->weights
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        else:
            self.net.apply(self.init_weights)
            print("Model Initializing")
            print("args.ctx: ", args.ctx)

        # summary
        if args.summary:
            summary(self.net, input_size=(3, args.crop_size, args.crop_size),
                    batch_dim=0, col_names=("input_size", "output_size", "num_params", "mult_adds"),
                    verbose=1)
        self.net.to(args.ctx[0])

        # loss
        self.criterion = SoftIoULoss()


        # lr_scheduler and optimizer
        optimizer_params = {'wd': args.weight_decay, 'learning_rate': args.lr, 'momentum': args.momentum}
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True
        if args.no_wd:   # impact of args.no_wd
            for key, value in self.net.params('.*beta|.*gamma|.*bias').items():
                value.wd_mult = 0.0

        # Adagrad
        self.optimizer = optim.Adagrad(self.net.parameters(), lr=optimizer_params['learning_rate'],
                                       weight_decay=optimizer_params['wd'])

        # SGD
        # self.optimizer = optim.SGD(self.net.parameters(), lr=optimizer_params['learning_rate'],
        #                            weight_decay=optimizer_params['wd'], momentum=optimizer_params['momentum'])

        # lr_scheduler with warm up
        # self.warm_up_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / args.warm_up_epochs)

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 - (epoch / args.epochs) ** 0.9)   # poly
        # self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)     # cosine



        ######### evaluation metrics #########
        self.score_thresh = args.score_thresh
        self.mIoU = mIoU(nclass=1)
        self.ROC = ROCMetric(nclass=1, bins=10)
        self.nIoU = nIoU(nclass=1)


        self.best_metric = 0
        self.best_iou = 0
        self.best_nIoU = 0



        ######## Training detail #########
        # epochs, max_iters = args.epochs, args.max_iters
        # log_per_iters, val_per_iters = args.log_iter, args.val_epoch * args.iters_per_epoch
        # save_per_iters = args.save_epoch * args.iters_per_epoch

    def training(self, epoch):

        train_loss = 0.0

        tbar = tqdm(self.train_data)

        self.net.train()
        for i, (images, labels) in enumerate(tbar):
            torch.cuda.empty_cache()

            images = images.to(args.ctx[0])
            labels = labels.to(args.ctx[0])

            self.optimizer.zero_grad()
            outputs = self.net(images)
            loss = self.criterion(outputs.squeeze(1), labels.float())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()    # loss.item():mean loss of this batch
            self.lr_scheduler.step()

            tbar.set_description('Epoch: %d || Iters: %d || Training loss: %.4f'
                                 % (epoch, (epoch * len(self.train_data)) + i, train_loss))

        train_logger.info("Epoch:{:d} ; Training loss:{:.4f}".format(epoch, train_loss / len(self.train_data)))


    def validation(self, epoch):

        tbar = tqdm(self.eval_data)

        self.mIoU.reset()
        self.nIoU.reset()


        val_loss = 0.0

        self.net.eval()
        for i, (images, labels) in enumerate(tbar):
            images = images.to(args.ctx[0])
            labels = labels.to(args.ctx[0])
            with torch.no_grad():
                outputs = self.net(images)

            loss = self.criterion(outputs.squeeze(1), labels.float())

            val_loss += loss.item()

            # metirc - first move to cpu

            self.mIoU.update(outputs.cpu(), labels.cpu())
            self.nIoU.update(outputs.cpu(), labels.cpu())
            self.ROC.update(outputs.cpu(), labels.cpu())
            ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
            _, mIoU = self.mIoU.get()
            _, nIoU = self.nIoU.get()

            tbar.set_description('Epoch %d || Iters: %d || val_loss: %.4f || mIoU: %.4f || nIoU: %.4f'
                                % (epoch, (epoch * len(self.eval_data)) + i,
                                   val_loss / len(self.eval_data), mIoU, nIoU))
        val_logger.info("Epoch{:d} ; val_loss:{:.4f} ; mIoU:{:.4f} ; nIoU:{:.4f} "
                        .format(epoch, val_loss / len(self.eval_data), mIoU, nIoU,))
        val_logger.info(f'precision:{precision} ; recall:{recall}')

        ## for all epochs
        if mIoU > self.best_iou:
            self.best_iou = mIoU

        if nIoU > self.best_nIoU:
            self.best_nIoU = nIoU

        if epoch >= args.epochs - 1:
            print("best_iou: ", self.best_iou)
            print("best_nIoU: ", self.best_nIoU)


    # def predict(self, visual_dir):
    #     tbar = tqdm(self.test_data)
    #     self.metric.reset()
    #
    #     for i, (images, labels) in enumerate(tbar):
    #         images = images.to(args.ctx[0])
    #         labels = labels.to(args.ctx[0])
    #         with torch.no_grad():
    #             self.net.load_state_dict(torch.load(args.predict_checkpoint))
    #             outputs = self.net(images.unsqueeze(0))
    #
    #
    #         # metirc  sigmoid + threshold for a batch images 4D
    #         self.metric.update(outputs, labels.unsqueeze(0))
    #
    #
    #         if args.visual_img:
    #             img = images[0].cpu().numpy()
    #             label = labels.cpu().numpy()
    #             output = nn.functional.sigmoid(outputs.squeeze(1)[0]).cpu().numpy()
    #             output = (output > self.score_thresh).astype(np.int64)
    #             plot_img_and_mask(img, label, output, visual_dir)
    #
    #
    #     pixAcc, mIoU, nIoU = self.metric.get()
    #     tbar.set_description('pixAcc: %.4f || IoU: %.4f || nIoU: %.4f'
    #                          % (pixAcc, mIoU, nIoU))
    #     test_logger.info("pixAcc:{:.4f}  mIoU:{:.4f}  nIoU:{:.4f}"
    #                     .format(pixAcc, mIoU, nIoU))


    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # if m.bias is not None:
            #     init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)




if __name__ == "__main__":
    args = parse_args()

    ######## enviroment ########
    if args.colab:
        args.data_root = '/content/experiments/data'
        args.log_dir = '/content/drive/MyDrive/experimentsresult/logs'
        args.save_dir = '/content/drive/MyDrive/experimentsresult/params'
        args.visual_dir = '/content/drive/MyDrive/experimentsresult/visual'

    # ######## training ########
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Total Epochs:', args.epochs)
        ## logger create
        train_log = '{}_{}_'.format(args.net_choice, args.dataset) + '_train_log.txt'
        train_logger = setup_logger("training process", args.log_dir, filename=train_log)
        train_logger.info("Using {} GPUs".format(len(args.ctx)))
        train_logger.info(args)
        val_log = '{}_{}_'.format(args.net_choice, args.dataset) + '_val_log.txt'
        val_logger = setup_logger("validation process", args.log_dir, filename=val_log)
        val_logger.info("Using {} GPUs".format(len(args.ctx)))
        val_logger.info(args)
        for epoch in range(args.epochs):
            trainer.training(epoch)
            save_checkpoint(trainer.net, args, epoch)
            if not args.no_val:
                trainer.validation(epoch)

        # visualize training and eval metrics
        if os.path.exists(os.path.join(args.log_dir, train_log)):
            train_visualize(os.path.join(args.log_dir, train_log), args.visual_dir)
        else:
            raise ValueError("train_log is not found")

        if os.path.exists(os.path.join(args.log_dir, val_log)):
            val_visualize(os.path.join(args.log_dir, val_log), args.visual_dir)
        else:
            raise ValueError("val_log is not found")

    # if args.predict:
    #     test_log = '{}_'.format(os.path.basename(args.predict_checkpoint).split('.')[0]) + 'epoch_test_log.txt'
    #     test_logger = setup_logger("testing process", args.log_dir, filename=test_log)
    #     trainer.predict(args.visual_dir)








