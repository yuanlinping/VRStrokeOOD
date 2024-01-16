# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#coding=utf-8
import torch
import torch.nn as nn
from network import act_network

def get_fea(args,onlyxyz = False):
    # net=act_network.ActNetwork(args.dataset)
    if onlyxyz:
        args.dataset = 'stroke_onlyxyz'
    net = act_network.RNN(args.dataset)
    return net

def accuracy(args,network, loader, weights,usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            all_x = data[0].cuda().float()
            if(args.onlyxyz):
                all_x = all_x[:,0:2,:,:]
            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            y = data[5].cuda().float()
            if usedpredict=='p':
                p = network.predict(all_x_noise)
            else:
                p=network.predict1(all_x_noise)
            test_loss = nn.functional.mse_loss(p, y)
            # if weights is None:
            #     batch_weights = torch.ones(len(x))
            # else:
            #     batch_weights = weights[weights_offset : weights_offset + len(x)]
            #     weights_offset += len(x)
            # batch_weights = batch_weights.cuda()
            # if p.size(1) == 1:
            #     correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            # else:
            #     correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += test_loss
    network.train()

    return total

def acc_class(args,network, loader, weights,usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for data in loader:
            all_x = data[0].cuda().float()
            if(args.onlyxyz):
                all_x = all_x[:,0:2,:,:]
            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            y = data[1].cuda().float()
            p = network.predict2(all_x_noise)
            
            # Convert probabilities to predicted labels
            predicted_labels = torch.argmax(p, dim=1)
            
            # Count number of correct predictions
            correct += (predicted_labels == y).sum().item()
            
            # Count total number of samples
            total += y.size(0)

        # Calculate accuracy
        accuracy = correct / total
        print("Accuracy: {:.2%}".format(accuracy))
    network.train()

    return accuracy

def acc_disc(network, loader, weights,usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for data in loader:
            all_x = data[0].cuda().float()
            
            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            y = data[1].cuda().float()
            p = network.predict3(all_x_noise)
            
            # Convert probabilities to predicted labels
            predicted_labels = torch.argmax(p, dim=1)
            
            # Count number of correct predictions
            correct += (predicted_labels == y).sum().item()
            
            # Count total number of samples
            total += y.size(0)

        # Calculate accuracy
        accuracy = correct / total
        print("Accuracy: {:.2%}".format(accuracy))
    network.train()

    return accuracy