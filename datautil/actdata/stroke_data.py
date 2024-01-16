# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8

from datautil.actdata.util import *
from datautil.util import mydataset, Nmax
import numpy as np
import torch


class StrokeList(mydataset):
    def __init__(self, args, dataset, root_dir, group_list, group_num, transform=None, target_transform=None, pclabels=None, pdlabels=None, shuffle_grid=True, eval_mode = "user_domain"):
        super(StrokeList, self).__init__(args)
        self.domain_num = 0
        self.dataset = dataset
        self.task = 'stroke_data'
        self.transform = transform
        self.target_transform = target_transform
        x, cy, py, sy,z = loaddata_from_numpy(self.dataset, self.task, root_dir,args.zclass_random,args.shapeclass_random)
        print(f"args.zclass_random:{args.zclass_random}")
        print(x.shape)
        self.group_list = group_list
        if eval_mode == 'user_domain':
            self.comb_position(x, cy, py, sy,z)
        elif eval_mode == 'shape_domain':
            self.comb_shape(x, cy, py, sy,z)

        self.x = self.x[:, :, np.newaxis, :]
        self.transform = None
        self.x = torch.tensor(self.x).float()
        self.z = torch.tensor(self.z).float()
        self.values = self.z
        if pclabels is not None:
            self.pclabels = pclabels
        else:
            self.pclabels = np.ones(self.labels.shape)*(-1)
        if pdlabels is not None:
            self.pdlabels = pdlabels
        else:
            self.pdlabels = np.ones(self.labels.shape)*(0)
    def comb_shape(self,x, cy, py, sy,z):
        for i, shape in enumerate(self.group_list):
            index = np.where(sy == shape)[0]
            tx,tz, tcy, tsy = x[index], z[index],cy[index], sy[index]
            if i == 0:
                self.x, self.z,self.labels,self.dlabels= tx, tz,tcy,tsy
            else:
                self.x = np.vstack((self.x, tx))
                self.z = np.vstack((self.z, tz))
                self.labels = np.hstack((self.labels, tcy))
                self.dlabels = np.hstack((self.dlabels, tsy))
    def comb_position(self, x, cy, py, sy,z):
        for i, peo in enumerate(self.group_list):
            index = np.where(py == peo)[0]
            tx,tz, tcy, tsy = x[index], z[index],cy[index], sy[index]
            if i == 0:
                self.x, self.z,self.labels,self.dlabels= tx, tz,tcy,tsy
            else:
                self.x = np.vstack((self.x, tx))
                self.z = np.vstack((self.z, tz))
                self.labels = np.hstack((self.labels, tcy))
                self.dlabels = np.hstack((self.dlabels, tsy))