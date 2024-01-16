# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import numpy as np
from torch.utils.data import DataLoader
import torch
import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset

import datautil.actdata.stroke_data as stroke_data
task_act = {'stroke_data':stroke_data}


    

def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_stroke_dataloader(args):
    source_dataset_list = []  # Data for training
    target_data_list = []    # Data for testing
    pcross_act = task_act[args.task]
    
    if args.eval_mode == 'user_domain':  # Divide domains based on user for evaluation
        tmpp = args.act_people[args.dataset]  # Domain groups: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9)
        args.domain_num = len(tmpp)  # Number of domains
        for i, item in enumerate(tmpp):
            tdata = None
            if args.task == "stroke_data":
                tdata = pcross_act.StrokeList(  # Load and obtain StrokeList for training
                    args, args.dataset, args.data_dir, item, i, transform=actutil.act_train(), eval_mode="user_domain")
            if i in args.test_envs:  # For domains to be used as test sets
                print(f"user {item[0]} and {item[1]} is set as target data")
                target_data_list.append(tdata)
            else:  # For domains to be used for training
                source_dataset_list.append(tdata)
                if len(tdata) / args.batch_size < args.steps_per_epoch:
                    args.steps_per_epoch = len(tdata) / args.batch_size
    elif args.eval_mode == 'shape_domain':  # Divide domains based on shape
        tmpp = args.shape_domain[args.dataset]  # Domain groups: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [...], [...], [...]]
        args.domain_num = len(tmpp)
        for i, item in enumerate(tmpp):
            tdata = None
            if args.task == "stroke_data":
                tdata = pcross_act.StrokeList(
                    args, args.dataset, args.data_dir, item, i, transform=actutil.act_train(), eval_mode="shape_domain")
            if i in args.test_envs:  # For domains to be used as test sets
                print(f"shape {item} is set as target data")
                target_data_list.append(tdata)
            else:  # For domains to be used for training
                source_dataset_list.append(tdata)
                if len(tdata) / args.batch_size < args.steps_per_epoch:
                    args.steps_per_epoch = len(tdata) / args.batch_size
    elif args.eval_mode == 'infer2d':  # For cases where 2D data is used to generate 3D data (inference only)
        print("infer2d mode")
        tmpp = args.shape_domain[args.dataset]  # 0-10, which means only one domain in this case
        tdata = None
        for i, item in enumerate(tmpp):  # This loop is executed only once in this case
            if args.task == "stroke_data":
                tdata = pcross_act.StrokeList(
                    args, args.dataset, args.data_dir, item, i, transform=actutil.act_train(), eval_mode="shape_domain")
                target_data_list.append(tdata)
            if len(tdata) / args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata) / args.batch_size
        target_data = combindataset(args, target_data_list)  # Process this class even though there is only one domain
        l = len(target_data.labels)
        print(f"target dataset num: {l}")  # Print the total count; no division into test and train sets in this case
        train_loader = DataLoader(dataset=target_data, batch_size=args.batch_size,
                                  num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
        return train_loader
    rate = 0.2  # Ratio for validation set within the training set; not related to the test set
    args.steps_per_epoch = int(args.steps_per_epoch * (1 - rate))
    tdata = combindataset(args, source_dataset_list)
    l = len(tdata.labels)  # 3223
    print(f"train dataset num: {l}")
    
    # Train normalization
    print("use min-max normalization")  # Normalize the data below
    min_value = torch.min(tdata.values)
    max_value = torch.max(tdata.values)
    tdata.values = (tdata.values - min_value) / (max_value - min_value)
    index_all = np.arange(l)
    np.random.seed(args.seed)
    np.random.shuffle(index_all)
    ted = int(l * rate)
    index_tr, index_val = index_all[ted:], index_all[:ted]  # Randomly select validation and training sets
    tr = subdataset(args, tdata, index_tr)
    val = subdataset(args, tdata, index_val)
    target_data = combindataset(args, target_data_list)  # len: 1704
    l = len(target_data.labels)
    print(f"test dataset num: {l}")

    target_data.values = (target_data.values - min_value) / (max_value - min_value)

    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, tr, val, target_data)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, target_data, min_value, max_value

