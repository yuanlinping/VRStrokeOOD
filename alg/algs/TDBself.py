# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#coding=utf-8

from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from alg.modelopera import get_fea
from network import Adver_network,common_network
from alg.algs.base import Algorithm
from loss.common_loss import Entropylogits
from torch.utils.tensorboard import SummaryWriter  
class TDBself(Algorithm):
    '''
    update_a: 这个函数更新所有域上的特征提取器网络的参数，该函数的目的是让所有域的数据通过特征提取器网络映射到一个相同的特征空间中，从而降低不同域之间的差异性。具体而言，该函数的输入为一个数据batch，其输出为一个字典，包含了域特征提取器的损失。这个函数的作用是让特征提取器网络学习到一个对于所有域数据都有用的特征映射函数。

    update_d: 这个函数更新域分类器和域鉴别器的参数，通过域分类器和域鉴别器的学习，可以让特征提取器在域适应任务中具有更好的泛化性能。具体而言，该函数的输入为一个数据batch，其输出为一个字典，包含了域分类器和域鉴别器的损失。这个函数的作用是通过域分类器和域鉴别器的学习，降低不同域之间的差异性，从而提升模型在域适应任务中的性能。

    update: 这个函数在update_a和update_d之间切换执行，用于在特征提取器和分类器之间进行交替训练，从而提高域适应任务的性能。具体而言，该函数的输入为一个数据batch，其输出为一个字典，包含了所有损失的信息。该函数的作用是利用交替训练的方式，不断地优化特征提取器和分类器，以提高域适应任务的性能。
    '''
    def __init__(self, args):

        super(TDBself, self).__init__(args)

        self.onlyxyz = args.onlyxyz;
        self.featurizer = get_fea(args,self.onlyxyz)
        self.dbottleneck=common_network.feat_bottleneck(self.featurizer.in_features,args.bottleneck,args.layer)
        self.ddiscriminator = Adver_network.Discriminator(args.bottleneck,args.dis_hidden,args.num_classes)

        self.bottleneck=common_network.feat_bottleneck(self.featurizer.in_features,args.bottleneck,args.layer)
        self.classifier = common_network.feat_classifier(args.num_classes,args.bottleneck,args.classifier)
        self.regressor = common_network.feat_regressor(args.bottleneck,64)
        self.cgan_discriminator = common_network.cgan_discriminator(64,2,model_name="CNN")
        self.discriminator = Adver_network.Discriminator(args.bottleneck,args.dis_hidden,args.latent_domain_num)

        self.abottleneck=common_network.feat_bottleneck(self.featurizer.in_features,args.bottleneck,args.layer)
        #args.num_classes = 6;  args.latent_domain_num = 5
        self.aclassifier=common_network.feat_classifier(args.num_classes*args.latent_domain_num,args.bottleneck,args.classifier)
        self.dclassifier = common_network.feat_classifier(args.latent_domain_num,args.bottleneck,args.classifier)
        self.args=args
        self.BCEWithLogitsLoss= nn.BCEWithLogitsLoss()
        self.predict_mode = args.predict
        
        if self.predict_mode == False:
            from datetime import datetime
            TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
            self.writer = SummaryWriter('./log/' + TIMESTAMP)
        self.cnt = 0
        
        
    def update_d(self,minibatch,opt):#这是Diversify的第二步，首先进行对抗网络的训练，得到类不变的信息
        all_x=minibatch[0].cuda().float()
        if(self.onlyxyz):
            all_x = all_x[:,0:2,:,:]
        all_d1=minibatch[1].cuda().long()#这里是class
        all_c1=minibatch[4].cuda().long()#这里是pd
        all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
        all_x_noise = torch.cat([all_noise,all_x],dim=1)
        z1=self.dbottleneck(self.featurizer(all_x_noise))
        #这里梯度反转的作用是让ddiscriminator能够分辨class的同时，让特征提取器和dbottleneck提取的特征使ddiscriminator不能判断来自哪一类
        disc_in1=Adver_network.ReverseLayerF.apply(z1,self.args.alpha1)
        disc_out1=self.ddiscriminator(disc_in1)
        disc_loss=F.cross_entropy(disc_out1,all_d1,reduction='mean')
        cd1=self.dclassifier(z1)
        ent_loss=Entropylogits(cd1)*self.args.lam+F.cross_entropy(cd1,all_c1)
        loss=ent_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()    
        return {'total':loss.item(),'dis':disc_loss.item(),'ent':ent_loss.item()}

    def set_dlabel(self,loader):#重新划分domain label
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():#这一步不更新网络
            iter_test = iter(loader)
            for _ in range(len(loader)):
                # data = iter_test.next()
                data = next(iter_test)
                inputs = data[0]
                all_x = inputs.cuda().float()
                if(self.onlyxyz):
                    all_x = all_x[:,0:2,:,:]
                all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
                all_x_noise = torch.cat([all_noise,all_x],dim=1)
                index=data[-1]
                feas = self.dbottleneck(self.featurizer(all_x_noise))#通过特征提取器和bottlenet获取的特征
                #print(feas.shape)#torch.Size([160, 256])
                outputs = self.dclassifier(feas)#torch.Size([160, 5])#通过域分类器获取结果
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index=index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_index=np.hstack((all_index,index))
        all_output = nn.Softmax(dim=1)(all_output)
        
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        # print(all_fea.shape)#(4144, 257)
        # print(aff.shape)#(4144, 5)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])#每个域的质心
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        # print(initc.shape)(5, 257)
        # print(pred_label.shape)(4144,)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
    #在这里对pdlabel(pseudo domain label)进行更新
        loader.dataset.set_labels_by_index(pred_label,all_index,'pdlabel')
        print(Counter(pred_label))#Counter({0: 1753, 1: 905, 3: 845, 2: 513, 4: 128})
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generator_encoder(self,all_x):
        all_z = self.bottleneck(self.featurizer(all_x))#(160,256) 通过特征提取器和bottleneck提取出的结果
        all_preds_z = self.regressor(all_z)#(160,64) #通过回归器(也就是decoder)的结果
        return all_z,all_preds_z
    
    def domain_invariant_discriminator(self,disc_labels,all_z):#对抗网络，使特征没有域的区别
        disc_input = all_z
        disc_input=Adver_network.ReverseLayerF.apply(disc_input,self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_loss = F.cross_entropy(disc_out, disc_labels)
        return disc_loss
    
    def backward_cgan_discriminator(self,all_x,all_preds_z,all_value): #训练判别器的过程
        # Fake; 通过all_preds_stroke.detach()来避免前面几部分的反向传播
        all_preds_z = all_preds_z[:, np.newaxis, np.newaxis, :]#(160,1,1,64)
        all_preds_stroke = torch.cat((all_x[:,0:2,:,:],all_preds_z),dim=1)#(160,22,1,64)->(160,3,1,64)
        pred_fake = self.cgan_discriminator(all_preds_stroke.detach())#(160,2)
        fake_label = torch.zeros_like(pred_fake)#生成全为0的标签
        fake_dloss = self.BCEWithLogitsLoss(pred_fake,fake_label)
        # Real
        all_value = all_value[:, np.newaxis, np.newaxis, :]
        all_true_stroke = torch.cat((all_x[:,0:2,:,:],all_value),dim=1)
        pred_true = self.cgan_discriminator(all_true_stroke)
        real_label = torch.ones_like(pred_true)#生成全为1的标签
        real_dloss = self.BCEWithLogitsLoss(pred_true,real_label)
        #两个loss加起来
        d_loss = (fake_dloss + real_dloss)*0.5
        # print(f'dloss:{d_loss.item()}')
        if self.predict_mode == False:
            self.writer.add_scalar('loss/d_loss', d_loss.item(),  self.cnt)
        d_loss.backward()

    def backward_cgan_generator(self,all_x,all_preds_z,all_value):
        all_preds_z = all_preds_z[:, np.newaxis, np.newaxis, :]#(160,1,1,64)
        all_value = all_value[:, np.newaxis, np.newaxis, :]
        all_preds_stroke = torch.cat((all_x[:,0:2,:,:],all_preds_z),dim=1)#(160,22,1,64)->(160,3,1,64)
        pred_fake = self.cgan_discriminator(all_preds_stroke)#(160,1)
        fake_label = torch.ones_like(pred_fake)#生成全为1的标签来愚弄判别器
        fake_gloss = self.BCEWithLogitsLoss(pred_fake,fake_label)
        
        regression_loss = F.mse_loss(all_preds_z,all_value)

        return fake_gloss,regression_loss

    def update(self, data, opt,optdd):
        self.cnt = self.cnt + 1
        #####生成器部分#####
        #获取数据
        all_x = data[0].cuda().float()#(160,21,1,64)
        if(self.onlyxyz):
            all_x = all_x[:,0:2,:,:]
        all_y = data[1].cuda().long()
        disc_labels = data[4].cuda().long()
        all_value = data[5].cuda().float()#(160,64)
        #下面这步算作是encoder-decoder结构的generator
        all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
        all_x_noise = torch.cat([all_noise,all_x],dim=1)
        all_z, all_preds_z = self.generator_encoder(all_x_noise)#all_z是bottleneck的结果，all_preds_z是regressor最后生成的z坐标
        for i in range(1):
            #####先训练Discrimator，这种情况下需要冻结前面几个网络#####
            self.set_requires_grad(self.cgan_discriminator,True)
            optdd.zero_grad()
            self.backward_cgan_discriminator(all_x,all_preds_z,all_value)
            optdd.step()
        #####接下来训练生成器####
        opt.zero_grad()
        #冻结分类器
        self.set_requires_grad(self.cgan_discriminator,False)
        #获取生成loss
        fake_gloss, regression_loss =  self.backward_cgan_generator(all_x,all_preds_z,all_value)
        #这边是获取domain_invariant的步骤，也把他算作还是大的generator的一部分
        domain_disc_loss = self.domain_invariant_discriminator(disc_labels,all_z)
        ##重新加入分类器
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        generator_loss=regression_loss+ domain_disc_loss + 0.1*fake_gloss + classifier_loss 
        # generator_loss=domain_disc_loss + 0.1*fake_gloss + classifier_loss 
        if self.predict_mode == False:#在不预测的情况(训练)下，需要用tensorboard记录loss的变化
            # generator_loss=regression_loss+ domain_disc_loss  + classifier_loss 
            self.writer.add_scalar('loss/regression_loss', regression_loss.item(),  self.cnt)
            self.writer.add_scalar('loss/fake_gloss', fake_gloss.item(),  self.cnt)
            # self.writer.add_scalar('loss/domain_disc_loss', domain_disc_loss.item(),  self.cnt)
            self.writer.add_scalar('loss/classifier_loss', classifier_loss.item(),  self.cnt)
            
            # generator_loss=1*fake_gloss + regression_loss + classifier_loss
            self.writer.add_scalar('loss/generator_loss', generator_loss.item(),  self.cnt)
        generator_loss.backward()
        opt.step()
        return {'total':fake_gloss.item(),'class':regression_loss.item(),'dis':domain_disc_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].cuda().float() #(160,21,1,64)(batch_size,feature_num, , sample_num)
        if(self.onlyxyz):
            all_x = all_x[:,0:2,:,:]
        all_c = minibatches[1].cuda().long()#clable 这里对应的是样本所对应的分类(0-5)
        all_d = minibatches[4].cuda().long() #pdlabels 假的域标签，之前被初始化为0
        all_y=all_d*self.args.num_classes+all_c # s = d` ×C +y. all_y.shape (160)
        all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
        all_x_noise = torch.cat([all_noise,all_x],dim=1)
        all_z = self.abottleneck(self.featurizer(all_x_noise))#all_z(160,256)
        all_preds = self.aclassifier(all_z)#all_preds shape(160,30)即这里的30为 S = K × C = 5 * 6 = 30
        #第1次训练的时候all_y中的数值只为0-5，而第二次的时候就为0-29
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss=classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class':classifier_loss.item()}

    def predict(self, x):
        return self.regressor(self.bottleneck(self.featurizer(x)))

    def predict1(self,x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
    
    def predict2(self,x):
        return self.classifier(self.bottleneck(self.featurizer(x)))
    def predict3(self,x):
        return self.cgan_discriminator(self.regressor(self.bottleneck(self.featurizer(x))))
    def discriminate(self,x):
        return self.cgan_discriminator(self.regressor(self.bottleneck(self.featurizer(x))))