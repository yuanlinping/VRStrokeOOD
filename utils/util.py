# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append('eval%d_in' % i)
            eval_name_dict['valid'].append('eval%d_out' % i)
        else:
            eval_name_dict['target'].append('eval%d_out' % i)
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'TDBself': ['class', 'dis', 'total']}
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):
    args.select_position = {'emg': [0],'stroke':[0]}
    args.select_channel = {'emg': np.arange(8),'stoke': np.arange(2)}
    args.hz_list = {'emg': 1000,'stroke': 1000}
    args.act_people = {'emg': [[i*9+j for j in range(9)]for i in range(4)],'stroke': [[i*2+j for j in range(2)]for i in range(5)],'stroke2d': [[i*2+j for j in range(2)]for i in range(5)]}
    args.shape_domain = {'stroke':[[i*10+j for j in range(10)]for i in range(4)],'stroke2d':[[i*10+j for j in range(10)]for i in range(1)]}
    tmp = {'emg1': ((8, 1, 200), 6, 10),'stroke': ((2, 1, 64), 10, 10),'stroke2d': ((2, 1, 64), 10, 10),'stroke_quickdraw': ((2, 1, 64), 10, 10)}
    args.num_classes, args.input_shape, args.grid_size = tmp[
        args.dataset][1], tmp[args.dataset][0], tmp[args.dataset][2]

    return args


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--predict', action='store_true', help='enable predict mode')
    parser.add_argument('--quickdraw', action='store_true', help='predict quickdraw')
    parser.add_argument('--onlyxyz', action='store_true', help='when no time info,just use xyz data')
    parser.add_argument('--eval_mode', type=str, default="shape_domain",help = 'evaluation mode:[user_domain,shape_domain,infer2d]')
    parser.add_argument('--algorithm', type=str, default="TDBself")
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--alpha1', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=16, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='stroke')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--latent_domain_num', type=int, default=5)
    parser.add_argument('--local_epoch', type=int,
                        default=3, help='local iterations')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int,
                        default=250, help="max iterations")
    parser.add_argument('--model_size', default='median',
                        choices=['small', 'median', 'large', 'transformer'])
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--old', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default="stroke_data")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--zclass_random', action='store_true')
    parser.add_argument('--shapeclass_random', action='store_true')
    parser.add_argument('--output_dir', type=str, default=".")
    parser.add_argument('--weights', type=str, default="weight_epoch15.pt")
    parser.add_argument('--strokes_dir', type=str, default="results")
    args = parser.parse_args()
    args.steps_per_epoch = 10000000000
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args
