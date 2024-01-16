import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
# example of calculating the frechet inception distance
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.spatial.distance import directed_hausdorff

import numpy as np
from sklearn import metrics


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


    
# calculate frechet distance
def calculate_fd(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid



data_path='/home/linping/diversity-modified/evaluation/shape/shape0/results'
csvs = os.listdir(data_path)
real_strokes = []
fake_strokes = []
real_zs = []
fake_zs = []
for csv in csvs:
    csv_path = os.path.join(data_path,csv)
    data = np.loadtxt(csv_path,delimiter=',',skiprows=1)
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    out_z = data[:,3]
    stroke_real = np.concatenate((x[np.newaxis,:],y[np.newaxis,:],z[np.newaxis,:]))
    stroke_fake = np.concatenate((x[np.newaxis,:],y[np.newaxis,:],out_z[np.newaxis,:]))
    real_strokes.append(stroke_real) 
    fake_strokes.append(stroke_fake)
    real_zs.append(z)
    fake_zs.append(out_z)
real_zs = np.array(real_zs,dtype=float)
fake_zs = np.array(fake_zs,dtype=float)



fd = calculate_fd(real_zs, fake_zs)
print('FD (diff): %.3f' % fd)

hau = directed_hausdorff(real_zs, fake_zs)[0]
print(f'directed_hausdorff (same):{hau}')

mlinear=mmd_linear(real_zs, fake_zs)
mrbf = mmd_rbf(real_zs, fake_zs)
mpoly = mmd_poly(real_zs, fake_zs)
print(f"mmd_linear:{mlinear}")
print(f"mmd_rbf:{mrbf}")  
print(f"mmd_poly:{mpoly}")

print(fd)
print(hau)
print(mlinear)
print(mrbf)
print(mpoly)