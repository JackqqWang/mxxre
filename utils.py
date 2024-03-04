import copy
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100
import torch
import clip
import os
import sys
import os
import albumentations
from ChestXray_dataset import NIH_Dataset,NIH_binary_Dataset,NIH_segmentation_Dataset,NIH_segmentation_3D_Dataset,NCT_dataset

import sys
import cv2
import numpy as np
import os
from torch.utils.data import random_split
from sampling import chestxray_seprate,chestxray_seg_seprate
from flamby.datasets.fed_ixi import FedIXITiny






from flamby.datasets.fed_isic2019 import FedIsic2019

from torch.utils.data import ConcatDataset


def exp_details(args):
    print('\nExperimental details:')
    
    
    
    print(f'    Communication Rounds   : {args.communication_round}\n')
    print(f'    Number of users        : {args.num_users}')
    print(f'    Dataset      : {args.dataset}')
    


    print('    Federated parameters:')
    
    
    
    
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def convert_to_binary_mask(labels, class_index):
    """
    Convert categorical labels to a binary mask format.

    Args:
    labels (torch.Tensor): The tensor containing categorical labels.
    class_index (int): The index of the class to convert to the binary mask format.

    Returns:
    torch.Tensor: A binary mask where pixels belonging to `class_index` are 1, and others are 0.
    """
    
    
    binary_mask = (labels == class_index).float()  
    return binary_mask


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
class CustomCIFAR100(Dataset):
    def __init__(self, cifar_dataset, extra_matrix_VIT,extra_matrix_RN50):
        self.cifar_dataset = cifar_dataset
        self.extra_matrix_VIT = extra_matrix_VIT
        self.extra_matrix_RN50 = extra_matrix_RN50

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        extra_vector_VIT = self.extra_matrix_VIT[idx]
        extra_vector_RN50 = self.extra_matrix_RN50[idx]
        return image, label, extra_vector_VIT, extra_vector_RN50
class CustomSkinSeg(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path=dataset_path
        self.img_pathlist=[]
        self.label_pathlist=[]
        self.distribute_pathlist=[]
        for i in os.listdir(os.path.join(dataset_path,'data')):
            self.img_pathlist.append(i)
            self.label_pathlist.append(i.strip('.jpg')+'_segmentation.png')
            self.distribute_pathlist.append(i.strip('.jpg') + '_distrobute.pt')
        self.sz = 256
        
        self.augmentations = albumentations.Compose(
            [
                albumentations.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], always_apply=True),
            ]
        )
    def __len__(self):
        return len(self.img_pathlist)

    def __getitem__(self, idx):
        image=cv2.resize(cv2.imread(os.path.join(self.dataset_path,'data',self.img_pathlist[idx])),(self.sz,self.sz))
        label=cv2.imread(os.path.join(self.dataset_path,'label',self.label_pathlist[idx]))
        extra_vector=torch.load(os.path.join(self.dataset_path,'distribute',self.distribute_pathlist[idx]))
        if len(label.shape)==3:
            label = label[:, :, 0]
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        image=np.transpose(image, (2, 0, 1)).astype(np.float32)
        label = np.expand_dims(label, axis=0).astype(np.float32)
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            extra_vector[0]
        )

def get_public_dataset(args,dst_name=None):
    
    if dst_name=='3DSeg':
        dataset = NIH_segmentation_3D_Dataset(
            imgpath='/data/jiaqi/FLARE4seg',
        )
        dataloader = DataLoader(dataset,
                                batch_size=args.pub_batch_size, shuffle=False)
        data_list = []
        for image, label, vector,mask in dataloader:
            data_list.append([image, label, vector,mask])
    if dst_name=='NCT_dataset':
        logits_path="FLamby/med_output_logits_ViTL14.pt"
        extra_matrix_VIT=torch.load(logits_path)
        logits_path="FLamby/med_output_logits_RN50.pt"
        extra_matrix_RN50=torch.load(logits_path)
        dataset = NCT_dataset(
            '/data/jiaqi/NCT',extra_matrix_VIT,extra_matrix_RN50,train=False
        )
        custom_dataloader = DataLoader(dataset,
                                batch_size=args.pub_batch_size, shuffle=False)
        data_list = []
        for image, label, extra_vector_VIT,extra_vector_RN50 in custom_dataloader:
            data_list.append([image,label,extra_vector_VIT,extra_vector_RN50])

        
        
        
        
        
        
        
        
        
            
            

    if dst_name=='cifar':
        logits_path="FLamby/output_logits.pt"
        extra_matrix_VIT=torch.load(logits_path)
        logits_path="FLamby/output_logits_RN50.pt"
        extra_matrix_RN50=torch.load(logits_path)
        _, preprocess = clip.load('ViT-L/14', "cuda")
        cifar100_test = CIFAR100(root=os.path.expanduser("/data/jiaqi/CIFAR100"), train=False, transform=preprocess, download=True)
        custom_dataset = CustomCIFAR100(cifar100_test, extra_matrix_VIT, extra_matrix_RN50)
        custom_dataloader = DataLoader(custom_dataset, batch_size=args.pub_batch_size, shuffle=True)
        data_list=[]
        
        for image, label, extra_vector_VIT,extra_vector_RN50 in custom_dataloader:
            data_list.append([image,label,extra_vector_VIT,extra_vector_RN50])
    if dst_name=='skinseg':
        dataset_path='/data/jiaqi/public_seg'
        custom_dataset = CustomSkinSeg(dataset_path)
        custom_dataloader = DataLoader(custom_dataset, batch_size=args.pub_batch_size, shuffle=True)
        data_list = []
        for image, label, vector in custom_dataloader:
            data_list.append([image, label, vector])
    
    return data_list
def get_datasets(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    
        
    if args.dataset == 'ChestXray_seg':
        torch.manual_seed(42)
        dataset = NIH_segmentation_Dataset(
            imgpath='/data/jiaqi/Lungseg',
        )

        total_size = len(dataset)
        
        test_size = int(0.1 * total_size)
        train_size = total_size - test_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        
        
        
        
        
        
        
        
        
        user_groups = chestxray_seg_seprate(train_dataset, test_dataset, args.num_users)
        
        
        return train_dataset, test_dataset, user_groups

    if args.dataset == 'ChestXray_binary':
        train_dataset = NIH_binary_Dataset(
            imgpath='/data/jiaqi/chestxray_phe/chest_xray/train',
        )
        test_dataset = NIH_binary_Dataset(
            imgpath='/data/jiaqi/chestxray_phe/chest_xray/test',
        )
        
        

        user_groups = chestxray_seprate(train_dataset, test_dataset, args.num_users)
        
        
        return train_dataset, test_dataset, user_groups
    if args.dataset=='Fed_IXI':
        train_dataset_lst=[]
        seperate_mark_list = [0]
        for i in range(3):
            dst=FedIXITiny(transform=None, center=i, train=True, pooled=False)
            train_dataset_lst.append(dst)
            seperate_mark_list.append(len(dst)+seperate_mark_list[-1])
        train_dataset = ConcatDataset(train_dataset_lst)
        test_dataset_lst = []
        seperate_mark_list_test = [0]
        for i in range(3):
            dst=FedIXITiny(transform=None, center=i, train=False, pooled=False)
            test_dataset_lst.append(dst)
            seperate_mark_list_test.append(len(dst)+seperate_mark_list_test[-1])
        test_dataset = ConcatDataset(test_dataset_lst)
        user_groups = {}
        
        
       
        for i in range(3):
            user_groups[i] = [[i for i in range(seperate_mark_list[i], seperate_mark_list[i + 1])],
                              [i for i in range(seperate_mark_list_test[i], seperate_mark_list_test[i + 1])]]
        
        
        
        
        
        
        
            

    if args.dataset == 'ChestXray':
    
        train_dataset = NIH_Dataset(
            imgpath='/home/jmw7289/fed_med/data/NIH/images-224',
            csvpath='/home/jmw7289/fed_med/data/NIH/train_csv.csv',
            bbox_list_path='/home/jmw7289/fed_med/data/NIH/BBox_List_2017.csv',
            views = ["*"],
        )
        test_dataset = NIH_Dataset(
            imgpath='/home/jmw7289/fed_med/data/NIH/images-224',
            csvpath='/home/jmw7289/fed_med/data/NIH/test_csv.csv',
            bbox_list_path='/home/jmw7289/fed_med/data/NIH/BBox_List_2017.csv',
            views=["*"],
        )

        user_groups = chestxray_seprate(train_dataset,test_dataset, args.num_users)
        
        
        
        return train_dataset, test_dataset, user_groups
        
        

    if args.dataset == 'ISIC':
        train_dataset_lst=[]
        seperate_mark_list=[0]
        for i in range(6):
            dst=FedIsic2019(center=i, train=True)
            train_dataset_lst.append(dst)
            seperate_mark_list.append(len(dst)+seperate_mark_list[-1])
        train_dataset = ConcatDataset(train_dataset_lst)
        test_dataset_lst = []
        seperate_mark_list_test = [0]
        for i in range(6):
            dst = FedIsic2019(center=i, train=False)
            test_dataset_lst.append(dst)
            seperate_mark_list_test.append(len(dst) + seperate_mark_list_test[-1])
        test_dataset = ConcatDataset(test_dataset_lst)
        user_groups={}
        
        
        for i in range(6):
            user_groups[i]=[[i for i in range(seperate_mark_list[i],seperate_mark_list[i+1])],[i for i in range(seperate_mark_list_test[i],seperate_mark_list_test[i+1])]]


    return train_dataset, test_dataset, user_groups
