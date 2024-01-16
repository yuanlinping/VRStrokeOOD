# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
from torchvision import transforms
import numpy as np

def act_train():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def loaddata_from_numpy(dataset='dsads', task='cross_people', root_dir='./data/act/', zclass_random = False,shapeclass_random = False):
    if dataset == 'pamap' and task == 'cross_people':
        x = np.load(root_dir+dataset+'/'+dataset+'_x1.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y1.npy')
    elif dataset == 'stroke' and task == 'stroke_data':
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        z = np.load(root_dir+dataset+'/'+dataset+'_z.npy')
        py = np.load(root_dir+dataset+'/'+dataset+'_person.npy')
        if zclass_random:
            cy = np.load(root_dir+dataset+'/'+dataset+'_zrandom.npy')
        else:
            cy = np.load(root_dir+dataset+'/'+dataset+'_zclass.npy')
        if shapeclass_random:
            sy = np.load(root_dir+dataset+'/'+dataset+'_classrandom.npy')#class
        else:
            sy = np.load(root_dir+dataset+'/'+dataset+'_class.npy')#class
        t = np.load(root_dir+dataset+'/'+dataset+'_t.npy')

        x = np.concatenate((x, t[:, np.newaxis, :]), axis=1)
        return x,cy,py,sy,z
    elif dataset == 'stroke2d' and task == 'stroke_data':
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        z = np.load(root_dir+dataset+'/'+dataset+'_t.npy')
        py = np.load(root_dir+dataset+'/'+dataset+'_person.npy')
        cy = np.load(root_dir+dataset+'/'+dataset+'_zrandom.npy')
        sy = np.load(root_dir+dataset+'/'+dataset+'_class.npy')#class
        t = np.load(root_dir+dataset+'/'+dataset+'_t.npy')

        x = np.concatenate((x, t[:, np.newaxis, :]), axis=1)
        return x,cy,py,sy,z
    else:
        x = np.load(root_dir+dataset+'/'+dataset+'_x.npy')
        ty = np.load(root_dir+dataset+'/'+dataset+'_y.npy')
        cy, py, sy = ty[:, 0], ty[:, 1], ty[:, 2]
        return x, cy, py, sy
