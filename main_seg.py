
import os

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from tools import *
from server import *
from models import UNet
from tools import *


from options import args_parser
from utils import exp_details, get_datasets,get_public_dataset, average_weights
from update import LocalUpdate
from magic_tool import *
from resnet_family import *
from sampling import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


if __name__ == '__main__':
    start_time = time.time()


    args = args_parser()
    exp_details(args)
    
    device = args.device
    


    train_dataset, test_dataset, user_groups = get_datasets(args)


    if args.public==1:
        public_data_list=get_public_dataset(args,dst_name='NCT_dataset')
 


    large_client_idx = ranking_dict(user_groups)[:args.top_n]
    print("large client idx is: {}".format(large_client_idx))

    if args.dataset == 'ISIC' or args.dataset == 'ChestXray_binary':
        if args.small_model == 'ResNet20':
            global_model = resnet20()
            
            g_num_ftrs = global_model.linear.in_features
            global_model.linear = nn.Linear(g_num_ftrs, args.num_classes)
        if args.ours:
            if args.large_model == 'ResNet110':
                large_model = resnet110()
      
                l_num_ftrs = large_model.linear.in_features
                large_model.linear = nn.Linear(l_num_ftrs, args.num_classes)
        else:
            large_model = resnet20()
       
            l_num_ftrs = large_model.linear.in_features
            large_model.linear = nn.Linear(l_num_ftrs, args.num_classes)
    if args.dataset == 'ChestXray_seg':
        global_model = UNet(n_class = 1)


    global_weights = global_model.state_dict()
    train_loss, test_accuracy = [], []
  
    test_commmunication_acc = []
    
    print_every = 1
    with tqdm(total= args.communication_round, desc="Training Progress") as pbar:
        for epoch in range(args.communication_round):
            local_weights, local_losses = [], []
            print(f'\n | communication round : {epoch+1} |\n')
            global_model.train()
            
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            
            epoch_user_test_score = []
            for idx in idxs_users:
                if args.test_pass == 0:
                    if idx in large_client_idx:
                        print("start large client {}".format(idx))
                        large_model.train()
                        local_model = LocalUpdate(args=args, dataset=[train_dataset,test_dataset],
                                                idxs= user_groups[idx], small_model = global_model)
             
                        w, loss = local_model.large_client_update_weights(large_model=large_model,
                            small_model=copy.deepcopy(global_model), global_round=epoch)
                
                        acc, test_loss = local_model.inference(model=global_model)
                        print("user_{}_test_acc_is: {}".format(idx, test_loss))
                        local_weights.append(copy.deepcopy(w))
                        local_losses.append(copy.deepcopy(loss))
 
                    
                    else:
                        print("start small client {}".format(idx))
            
                        local_model = LocalUpdate(args=args, dataset=[train_dataset,test_dataset],
                                                idxs= user_groups[idx], small_model = global_model)
              
                        w, loss = local_model.small_client_update_weights_final(
                                small_model=copy.deepcopy(global_model), global_round=epoch,public_data=public_data_list)
                        
                        acc, test_loss = local_model.inference(model=global_model)
                        print("user_{}_test_acc_is: {}".format(idx, test_loss))
                        local_weights.append(copy.deepcopy(w))
                        local_losses.append(copy.deepcopy(loss))

                    epoch_user_test_score.append(test_loss)
                test_commmunication_acc.append(sum(epoch_user_test_score)/len(epoch_user_test_score))

            
            print("start global avg")
            global_weights = average_weights(local_weights)

            
            global_model.load_state_dict(global_weights)
            global_model.to(device)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)


            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=[train_dataset,test_dataset],
                                        idxs=user_groups[idx],small_model = global_model)
                if args.test_pass == 0:
                    if args.dataset == 'ChestXray_seg':
                        acc, loss = local_model.seg_inference(model=global_model)
                
                    if c in large_client_idx:
        
                        acc, loss = local_model.inference(model=large_model)
                    else:
                        acc, loss = local_model.inference(model=global_model) 
                else:
                    avg_dice_score = local_model.seg_inference(model=global_model)
                list_acc.append(acc)
                list_loss.append(loss)
            test_accuracy.append(sum(list_acc)/len(list_acc))

            
            if (epoch+1) % print_every == 0:
                print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                print(f'Avg Local Training Loss : {np.mean(np.array(train_loss))}')
                print('Avg All Local test Accuracy: {:.2f}% \n'.format(100*test_accuracy[-1]))
        time.sleep(0.1)
        pbar.update(1)

    save_path = './save/{}_pub_{}'.format(args.dataset, args.public)
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
        print("The new directory is created!")


    torch.save(global_model.state_dict(), save_path + '/global_model_comm_{}.pth'.format(args.communication_round))
    with open(save_path + '/local_avg_test_acc_ours_{}_comm_{}.txt'.format(args.ours, args.communication_round), 'w') as filehandle:
        for listitem in test_accuracy:
            filehandle.write('%s\n' % listitem) 




    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
        



    plt.figure()
    plt.title('Local Average Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='r')
    plt.ylabel('Test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path + 'fed_test_accuracy_ours_{}_comm_{}.png'.format(args.ours, args.communication_round))

          

    





    
    



