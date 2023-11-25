
import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader

from data import IceContrast




from loss import SoftIoULoss
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR



class Trainer(object):
    def __init__(self, args):
        input_transform = transforms.Compose(
            [transforms.ToTensor(),
             # [.418, .447, .571], [.091, .078, .076] iceberg mean and std
             transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        ######## dataset and dataloader ########
        data_root = os.path.join(args.data_path, 'train')
        if os.path.exists(data_root) is False:
            raise FileNotFoundError("{} is not found".format(data_root))

        data_kwargs = {'base_size': args.base_size,
                       'crop_size': args.crop_size,
                       'transform': input_transform,
                       'root': data_root,
                       'base_dir': args.dataset}

        trainset = IceContrast(split=args.train_split, mode='train', **data_kwargs)
        valset = IceContrast(split=args.val_split, mode='testval', **data_kwargs)

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
            layers = [args.blocks] * 3
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
        self.criterion = SoftIoULoss()

        # lr_scheduler and optimizer
        lr_lambda = lambda epoch: 1 - (epoch / args.epochs) ** 0.9
        self.lr_scheduler =

        optimizer_params = {'wd': args.weight_decay,
                            'learning_rate': args.lr,}
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True
        if args.no_wd:
            for k, v in self.net.params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0  # ?

        self.optimizer = optim.Adagrad(self.net.parameters(), lr=optimizer_params['learning_rate'],
                                  weight_decay=optimizer_params['wd'])








def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.PReLU):
        init.constant_(m.weight, 0.25)
