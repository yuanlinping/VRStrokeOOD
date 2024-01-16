# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# coding=utf-8
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from alg.opt import *
from alg import alg, modelopera
from alg.algs.TDBself import TDBself
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_stroke_dataloader

def save_model(args, epoch: int, algorithm) -> None:
    """ Save the model for a given epoch.
    Args:
        epoch (int): Current epoch number.
        algorithm: The training algorithm object.
    """
    filename = f"weight_epoch{epoch}.pt"
    filename = os.path.join(args.output_dir,filename)
    if epoch > 0:
        os.remove(f"weight_epoch{epoch - 1}.pt")
    torch.save(algorithm.state_dict(), filename)
    print(f"Model saved in {filename}")

def mean_filter(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """ Apply a mean filter to an array.
    Args:
        arr (np.ndarray): Input array.
        window_size (int): Size of the moving window.
    Returns:
        np.ndarray: Array after applying mean filter.
    """
    result = np.zeros_like(arr)
    half_window = window_size // 2

    for i in range(half_window, len(arr) - half_window):
        result[i] = np.mean(arr[i - half_window: i + half_window + 1])

    for i in range(half_window):
        result[i] = np.mean(arr[: i + half_window + 1])
        result[-(i + 1)] = np.mean(arr[-(i + half_window + 1):])

    return result

def predict3d(args, target_loader, min_value, max_value) -> None:
    """ Perform prediction on 3D data.
    Args:
        args: Arguments for the prediction.
        target_loader: Data loader for the target dataset.
        min_value: Minimum value for normalization.
        max_value: Maximum value for normalization.
    """
    # output_3D = True
    output_3D = False # Flag to save 3D model
    output_stroke = True  # Flag to output stroke data
    if output_3D:
        if not os.path.exists('output_3D'):
             # Create output directory if it doesn't exist
            os.makedirs('output_3D')
            print("output_3D folder created!")
        else:
            print("output_3D folder already exists.")
    algorithm.load_state_dict(torch.load(args.weights))
    algorithm.eval()
    real_zs = None
    fake_zs =None
    with torch.no_grad():
        file_num = 0
        cnt = 0
        correct = 0
        total = 0
        first_time = 0
        for data in target_loader:
            all_x = data[0].cuda().float()
            if(args.onlyxyz):
                all_x = all_x[:,0:2,:,:]
            z_class =data[1].cuda().float() 
            y = data[5].cuda().float() 
            y = y*(max_value - min_value) + min_value 

            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            p = algorithm.predict(all_x_noise)#z value
            p = p*(max_value - min_value) + min_value#undo normalization
            print(p.shape)
            pl = algorithm.predict2(all_x_noise)#z classification
            predicted_labels = torch.argmax(pl, dim=1)
            correct += (predicted_labels == z_class).sum().item()

            if(first_time == 0):#for train fid
                real_zs = y
                fake_zs = p
                first_time = first_time + 1
            else:
                real_zs = torch.cat((real_zs,y),0)
                fake_zs = torch.cat((fake_zs,p),0)
            # print(real_zs.shape)
            
            # Count total number of samples
            total += y.size(0)
            x = torch.squeeze(all_x)
            # print(x.shape)
            for i in range(x.shape[0]):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                xi = x[i].cpu().numpy()[0,:]
                yi = x[i].cpu().numpy()[1,:]
                zi = y[i].cpu().numpy()
                ti = x[i].cpu().numpy()[-1,:]
                out_zi = p[i].cpu().numpy()
                out_zi = mean_filter(out_zi)
                out_zi = mean_filter(out_zi)
                # print(f'x1:{xi}')
                # print(f'y1:{yi}')
                # print(f'z1:{zi}')
                # print(f'out_zi:{out_zi}')
                ax.scatter(xi, yi, zi, c='b', marker='o',label='groudtruth')
                ax.scatter(xi, yi, out_zi, c='r', marker='o',label='prediction')
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                ax.legend() 
                if output_3D:
                    with open(f"./output_3D/test_{cnt}.pickle", 'wb') as f:
                        cnt = cnt + 1
                        pickle.dump(ax, f)
                        # print(f'save to ./output_3D/test_{cnt}.pickle')
                elif output_stroke:
                    output_dir = args.strokes_dir
                    if os.path.exists(output_dir) == False:
                        os.makedirs(output_dir)
                    file_num = file_num + 1
                    csv_filename = os.path.join(output_dir,str(file_num) + "_stroke.csv")#each file contains a stroke
                    out_data = np.vstack((xi,yi,zi,out_zi,ti))
                    np.savetxt(csv_filename, out_data.T,header='X,Y,Z,Pred_Z,TimestampInSecond', delimiter=',',comments='')
                    # print(out_data.T.shape)
                    # print(f"save to {csv_filename}")

                else:
                    plt.show()
                plt.close()
                    # Calculate accuracy
        accuracy = correct / total
        print("Accuracy: {:.2%}".format(accuracy))
        real_zs =  real_zs.cpu().numpy()
        fake_zs =  fake_zs.cpu().numpy()
        # np.savetxt('real_zs.csv',real_zs,delimiter=',')
        # np.savetxt('fake_zs.csv',fake_zs,delimiter=',')
        # print("save to real_zs.csv and fake_zs.csv")

def predict2d(args,target_loader): 
    # output_3D = True
    algorithm.load_state_dict(torch.load('evaluation/standard/algorithm_v8_64_best_standard.pt'))
    algorithm.eval()
    real_zs = None
    fake_zs =None
    with torch.no_grad():
        file_num = 0
        correct = 0
        total = 0
        for data in target_loader:
            all_x = data[0].cuda().float()
            if(args.onlyxyz):
                all_x = all_x[:,0:2,:,:]
            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            p = algorithm.predict(all_x_noise)
            shape_labels = data[2]
            print(shape_labels)
            print(p.shape)
            pl = algorithm.predict2(all_x_noise)
            predicted_zlabels = torch.argmax(pl, dim=1)

    
            x = torch.squeeze(all_x)
            # print(x.shape)

            for i in range(x.shape[0]):
                xi = x[i].cpu().numpy()[0,:]
                yi = x[i].cpu().numpy()[1,:]
                sl = np.tile(shape_labels[i].cpu().numpy(), 64).astype(int)
                ti = x[i].cpu().numpy()[-1,:]
                out_zi = p[i].cpu().numpy()
                out_zi = mean_filter(out_zi)
                out_zi = mean_filter(out_zi)
                # print(f'x1:{xi}')
                # print(f'y1:{yi}')
                # print(f'z1:{zi}')
                # print(f'out_zi:{out_zi}')
                output_dir = args.strokes_dir
                if os.path.exists(output_dir) == False:
                    os.makedirs(output_dir)
                file_num = file_num + 1
                csv_filename = os.path.join(output_dir,str(file_num) + "_stroke.csv")
                out_data = np.vstack((xi,yi,out_zi,ti,sl))
                np.savetxt(csv_filename, out_data.T,header='X,Y,Pred_Z,TimestampInSecond,ShapeLabel', delimiter=',',comments='')
                # print(out_data.T.shape)
                # print(f"save to {csv_filename
def predictQuickDraw(args): 
    # output_3D = True
    algorithm.load_state_dict(torch.load('algorithm_v8_onlyxyz.pt'))
    algorithm.eval()
    names = ['bowtie','cat','chair','clock','cup','envelope','eyeglasses','fish','postcard','tent']
    for name in names:
        sketches = np.load('data/quickdraw'+'/full_raw_'+name+'/sketches.npy')
        sketches = sketches[:,:,np.newaxis,:]
        with torch.no_grad():
            file_num = 0
            data = torch.from_numpy(sketches)
            all_x = data.cuda().float()
            all_xt = all_x
            all_x = all_x[:,0:2,:,:]
            all_noise = torch.randn(all_x.shape[0], all_x.shape[1], all_x.shape[2],all_x.shape[3]).to(all_x.device)
            all_x_noise = torch.cat([all_noise,all_x],dim=1)
            p = algorithm.predict(all_x_noise)
            print(p.shape)
            pl = algorithm.predict2(all_x_noise)
            predicted_zlabels = torch.argmax(pl, dim=1)


            x = torch.squeeze(all_x)
            # print(x.shape)
            xt = torch.squeeze(all_xt)
            for i in range(x.shape[0]):
                xi = x[i].cpu().numpy()[0,:]
                yi = x[i].cpu().numpy()[1,:]
                out_zi = p[i].cpu().numpy()
                out_zi = mean_filter(out_zi)
                out_zi = mean_filter(out_zi)
                ti = xt[i].cpu().numpy()[2,:]
                # print(f'x1:{xi}')
                # print(f'y1:{yi}')
                # print(f'z1:{zi}')
                # print(f'out_zi:{out_zi}')
                output_dir = os.path.join(args.strokes_dir,name)
                if os.path.exists(output_dir) == False:
                    os.makedirs(output_dir)
                file_num = file_num + 1
                csv_filename = os.path.join(output_dir,str(file_num) + "_stroke.csv")#每个文件包含一个stroke
                out_data = np.vstack((xi,yi,out_zi,ti))
                np.savetxt(csv_filename, out_data.T,header='X,Y,Pred_Z,TimeStamp', delimiter=',',comments='')
                # print(out_data.T.shape)
                # print(f"save to {csv_filename


if __name__ == '__main__':
    torch.backends.cudnn.enabled = False
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = get_args()

    s = print_args(args, [])
    set_random_seed(args.seed)

    print_environ()
    print(s)
    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num
    if args.quickdraw == False:
        if args.eval_mode == 'infer2d':
            target_loader = get_stroke_dataloader(args)
        else:
            train_loader, train_loader_noshuffle, valid_loader, target_loader, traindata, validdata, targetdata,min_value,max_value = get_stroke_dataloader(
                args)
    #data consists of: 0:x，1:ctarget，2:dtarget，3:pctarget(pseudo class)，4:pdtarget(pseudo domain)
    # algorithm_class = alg.get_algorithm_class(args.algorithm)# load model
    algorithm_class =TDBself
    algorithm = algorithm_class(args).cuda()
    if args.predict:
        if(args.quickdraw):
            predictQuickDraw(args)
        elif args.eval_mode == 'infer2d':#
            predict2d(args,target_loader)
        else:
            predict3d(args,target_loader,min_value,max_value)
    else:
        best_valid_acc, target_acc = 0, 0


        # algorithm.load_state_dict(torch.load('./algorithm_v8_3_best_all2.pt'))
        algorithm.train()
        optd = get_optimizer(algorithm, args, nettype='TDBADV')#for step 2
        opt = get_optimizer(algorithm, args, nettype='TDBCLS')#for step 3
        opta = get_optimizer(algorithm, args, nettype='TDBALL')#for step 1
        optdd = get_optimizer(algorithm,args,nettype='CGAND')#GAN disc
        # algorithm.load_state_dict(torch.load('algorithm_v2_7.pt'))
        best_mse = 9999
        best_round = -1
        for round in range(0,args.max_epoch,1):
            
            print('====round %d=====' % round)
            print('====start obtain all features====')
            # # Fine-grained Feature Update
            loss_list = ['class']
            print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)
            for step in range(args.local_epoch):
                for data in train_loader:
                    loss_result_dict = algorithm.update_a(data, opta)
                print_row([step]+[loss_result_dict[item]
                                for item in loss_list], colwidth=15)

            print('====start domain splitting training====')
            #Latent distribution characterization
            loss_list = ['total', 'dis', 'ent']
            print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)
            
            for step in range(args.local_epoch):
                for data in train_loader:
                    loss_result_dict = algorithm.update_d(data, optd)
                print_row([step]+[loss_result_dict[item]
                                for item in loss_list], colwidth=15)

            algorithm.set_dlabel(train_loader)

            print('====start DANN class training====')
            loss_list = alg_loss_dict(args)
            eval_dict = train_valid_target_eval_names(args)
            print_key = ['epoch']
            print_key.extend([item+'_loss' for item in loss_list])
            print_key.extend([item+'_mse' for item in eval_dict.keys()])
            print_key.append('total_cost_time')
            print_row(print_key, colwidth=15)

            last_results_keys = None
            sss = time.time()
            for step in range(args.local_epoch):
                step_start_time = time.time()
                for data in train_loader:
                    step_vals = algorithm.update(data, opt,optdd)

                results = {
                    'epoch': step,
                }

                results['train_mse'] = modelopera.accuracy( #mse
                    args,algorithm, train_loader_noshuffle, None)

                results['train_acc'] = modelopera.acc_class(#for z classification
                    args,algorithm, train_loader_noshuffle, None)

                results['valid_mse'] = modelopera.accuracy(args,algorithm, valid_loader, None)

                results['target_mse'] = modelopera.accuracy(args,algorithm, target_loader, None)
    
                for key in loss_list:
                    results[key+'_loss'] = step_vals[key]
                results['total_cost_time'] = time.time()-sss
                print_row([results[key] for key in print_key], colwidth=15)

            save_model(args, round,algorithm)
        print('target acc:%.4f' % target_acc)
#remote