


from options import args_parser
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from resnet_family import *
from seg_loss import *
from utils import *
args = args_parser()
device = args.device


class PrivateClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(PrivateClassifier, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x)
    

class PublicClassifier(nn.Module):
    def __init__(self, input_features, num_classes):
        super(PublicClassifier, self).__init__()
        self.linear = nn.Linear(input_features, num_classes)

    def forward(self, x):
        return self.linear(x)



class PrivateClassifier_seg(nn.Module):
    def __init__(self):
        super(PrivateClassifier_seg, self).__init__()
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        return  self.outconv(x)
    
class PublicClassifier_seg(nn.Module):
    def __init__(self):
        super(PublicClassifier_seg, self).__init__()
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        return  self.outconv(x)



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def student_ranking_loss_IXI(student_logits, teacher_logits, top_k=None):
    return 0

def student_ranking_loss(student_logits, teacher_logits, top_k=None):
    bottom_k = student_logits.shape[1] - top_k
    student_indices = torch.topk(student_logits, top_k, dim=1).indices
    all_indices = set(range(student_logits.shape[1]))
    bottom_indices = torch.tensor([[i for i in all_indices if i not in indices] for indices in student_indices]).to(student_logits.device)
    
    top_logits = torch.gather(teacher_logits, 1, student_indices)
    
    bottom_logits = torch.gather(teacher_logits, 1, bottom_indices)

    top_expanded = top_logits.unsqueeze(-1).expand(-1, -1, bottom_k)
    bottom_expanded = bottom_logits.unsqueeze(-2).expand(-1, top_k, -1)
    
    
    
    pairwise_losses = F.relu(top_expanded - bottom_expanded) / (top_k*2)
    
 
    loss = pairwise_losses.sum()
    return loss

distillation_criterion = nn.KLDivLoss() 
metrics = defaultdict(float)


def dice_coefficient(pred, target):
    """Compute the Dice coefficient."""
    smooth = 1.0  
 
    pred = pred.bool()
    target = target.bool()
    
 
    intersection = (pred & target).float().sum(dim=(2, 3)) 

    pred_sum = pred.float().sum(dim=(2, 3))
    target_sum = target.float().sum(dim=(2, 3))
    
  
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.mean()

class LocalUpdate(object):
  
    def __init__(self, args, dataset, idxs, small_model):
        self.args = args
        self.trainloader, self.testloader = self.train_val_test(
            dataset, idxs)
      
        self.criterion = nn.CrossEntropyLoss().to(device)
    def send_small_model_back_cpu(self, small_model):
        small_model.to('cpu')


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        
        idxs_train = idxs[0][:int(0.9*len(idxs[0]))]
        
        idxs_val = idxs[0][int(0.9*len(idxs[0])):int(1.0*len(idxs[0]))]
        idxs_test = idxs[1]

        trainloader = DataLoader(DatasetSplit(dataset[0], idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
       
        testloader = DataLoader(DatasetSplit(dataset[1], idxs_test),
                                batch_size=len(idxs_test), shuffle=False)


        return trainloader, testloader
    def large_client_update_weights(self, large_model, small_model, global_round):
      
        large_model.to(device)
        small_model.to(device)
     
        epoch_loss = []
        self.T = args.temperature

        T = self.T

      
        if self.args.optimizer == 'sgd':
            large_model_optimizer = torch.optim.SGD(large_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
            small_model_optimizer = torch.optim.SGD(small_model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            large_model_optimizer = torch.optim.Adam(large_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
            small_model_optimizer = torch.optim.Adam(small_model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                large_model.eval()
                small_model.train()
                large_model_optimizer.zero_grad()
                small_model_optimizer.zero_grad()

                for param in small_model.parameters():
                    param.requires_grad = True

                for param in large_model.parameters():
                    param.requires_grad = False
                if args.dataset == 'ChestXray_binary':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                if args.dataset == 'ChestXray':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                with torch.no_grad():
                    large_model_logits = large_model(images)
                small_model_logits = small_model(images)
                
                
                soft_labels = torch.softmax(large_model_logits / T, dim=1)
                
                loss_distillation = distillation_criterion(F.log_softmax(small_model_logits / T, dim=1),
                                                    soft_labels)
                
                
                
                ce_small_loss = self.criterion(small_model_logits, labels)
                total_small_loss = ce_small_loss + loss_distillation
                total_small_loss.backward()
                small_model_optimizer.step()


                
                large_model.train()
                small_model.eval()

                for param in small_model.parameters():
                    param.requires_grad = False

                for param in large_model.parameters():
                    param.requires_grad = True

                ce_large_loss = self.criterion(large_model_logits, labels)
                small_model_logits = small_model(images)
                large_model_logits = large_model(images)
                soft_labels = torch.softmax(large_model_logits / T, dim=1)
                with torch.no_grad():
                    ce_small_loss = self.criterion(small_model_logits, labels)
                if args.dataset == 'ChestXray_binary':
                    top_k_ranking = 1
                if args.dataset == 'ISIC':
                    top_k_ranking = 5
                rkd_loss = student_ranking_loss(small_model_logits, large_model_logits, top_k=top_k_ranking)
                loss_distillation = distillation_criterion(F.log_softmax(small_model_logits / T, dim=1),
                                                    soft_labels)
                total_large_loss = ce_large_loss + loss_distillation + 0.2 * rkd_loss
                total_large_loss.backward()
                large_model_optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), total_large_loss.item()))
                batch_loss.append(total_large_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return large_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def get_small_model(self):
        return self.small_model
    
    def small_client_update_weights_final(self, small_model, global_round, public_data):
        
        
        private_c = PrivateClassifier(input_features=64, num_classes=args.num_classes).to(device)
        public_c = PublicClassifier(input_features=64, num_classes=100).to(device)
        small_model.to(device)
        
        small_model.train()
        epoch_loss = []

        
        if self.args.optimizer == 'sgd':
            ce_optimizer = torch.optim.SGD(list(small_model.parameters()) + list(private_c.parameters()), lr=self.args.lr,
                                        momentum=0.5)
            kd_optimizer = torch.optim.SGD(list(small_model.parameters())+list(public_c.parameters()), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            ce_optimizer = torch.optim.Adam(list(small_model.parameters()) + list(private_c.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)
            kd_optimizer = torch.optim.Adam(list(small_model.parameters())+list(public_c.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            public_data=random.sample(public_data, len(public_data))
            if len(self.trainloader) >len(public_data):
                pickled_data = public_data.repeat(len(self.trainloader)//len(public_data)+1,1,1,1)
            else:
                pickled_data = random.sample(public_data, len(self.trainloader))
            
            
            

            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                if args.dataset == 'ChestXray':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                if args.dataset == 'ChestXray_binary':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                
                

                ce_optimizer.zero_grad()

                feature_vector_private = small_model(images, return_feature_vect=True)
                ce_log_probs_private = private_c(feature_vector_private)

                ce_loss = self.criterion(ce_log_probs_private, labels)
                ce_loss.backward()
                ce_optimizer.step()


                pub_batch=pickled_data[batch_idx]
                public_image=pub_batch[0].to(device)
                
                public_vector=pub_batch[2].to(device)
                kd_optimizer.zero_grad()

                feature_vector_pub = small_model(public_image, return_feature_vect=True)
                kd_log_probs = public_c(feature_vector_pub)

                
                kd_loss = distillation_criterion(F.log_softmax(kd_log_probs / 1, dim=1),
                                                    public_vector / 1) 
                kd_loss.backward()
                kd_optimizer.step()

                loss_total = ce_loss + 1* kd_loss 
                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss_total.item()))
                batch_loss.append(loss_total.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        small_model.outconv = private_c.linear

        return small_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    def small_client_update_weights_final_2(self, small_model, global_round, public_data):
        
        
        private_c = PrivateClassifier(input_features=64, num_classes=args.num_classes).to(device)
        public_c = PublicClassifier(input_features=64, num_classes=100).to(device)
        small_model.to(device)
        
        small_model.train()
        epoch_loss = []

        
        if self.args.optimizer == 'sgd':
            ce_optimizer = torch.optim.SGD(list(small_model.parameters()) + list(private_c.parameters()), lr=self.args.lr,
                                        momentum=0.5)
            kd_optimizer = torch.optim.SGD(list(small_model.parameters())+list(public_c.parameters()), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            ce_optimizer = torch.optim.Adam(list(small_model.parameters()) + list(private_c.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)
            kd_optimizer = torch.optim.Adam(list(small_model.parameters())+list(public_c.parameters()+ theta), lr=self.args.lr,
                                         weight_decay=1e-4)
        theta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        for iter in range(self.args.local_ep):
            a = theta
            b = 1 - a
            batch_loss = []
            public_data=random.sample(public_data, len(public_data))
            if len(self.trainloader) >len(public_data):
                pickled_data = public_data.repeat(len(self.trainloader)//len(public_data)+1,1,1,1)
            else:
                pickled_data = random.sample(public_data, len(self.trainloader))
            
            
            

            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                if args.dataset == 'ChestXray':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                if args.dataset == 'ChestXray_binary':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                
                

                ce_optimizer.zero_grad()

                feature_vector_private = small_model(images, return_feature_vect=True)
                ce_log_probs_private = private_c(feature_vector_private)

                ce_loss = self.criterion(ce_log_probs_private, labels)
                ce_loss.backward()
                ce_optimizer.step()


                pub_batch=pickled_data[batch_idx]
                public_image=pub_batch[0].to(device)
                
                pv1=pub_batch[2].to(device)
                pv2=pub_batch[3].to(device)
                kd_optimizer.zero_grad()

                feature_vector_pub = small_model(public_image, return_feature_vect=True)
                kd_log_probs = public_c(feature_vector_pub)

                
                kd_loss_1 = distillation_criterion(F.log_softmax(kd_log_probs / 1, dim=1),
                                                    pv1 / 1) 
                kd_loss_2 = distillation_criterion(F.log_softmax(kd_log_probs / 1, dim=1),
                                                    pv2 / 1) 
                kd_loss = a *  kd_loss_1 + b * kd_loss_2
                kd_loss.backward()
                kd_optimizer.step()

                loss_total = ce_loss + 1* kd_loss 
                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss_total.item()))
                batch_loss.append(loss_total.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        small_model.outconv = private_c.linear

        return small_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
    def large_client_update_weights_seg(self, large_model, small_model, global_round):
        
        large_model.to(device)
        small_model.to(device)

        epoch_loss = []
        
      
        large_optimizer = optim.Adam(filter(lambda p: p.requires_grad, large_model.parameters()), lr=1e-4)
        small_optimizer = optim.Adam(filter(lambda p: p.requires_grad, small_model.parameters()), lr=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                a_index, b_index, c_index = 0, 0, 0  
                my_image = images[a_index, b_index, c_index, :, :]
                my_mask = labels[a_index, b_index, c_index, :, :]
                plt.imshow(my_image.cpu(), cmap='viridis')
                image_filename = 'images.png'
                plt.savefig(image_filename)
                plt.close()
                plt.imshow(my_mask.cpu(), cmap='viridis')
                image_filename = 'mask.png'
                plt.savefig(image_filename)
                plt.close()
                

                binary_mask = convert_to_binary_mask(labels, class_index=1)
                if args.dataset == 'ChestXray_seg':
                    images = images.repeat(1,3,1,1)
                    images.to(device)

                for param in small_model.parameters():
                    param.requires_grad = True

                for param in large_model.parameters():
                    param.requires_grad = False

                with torch.no_grad():
                    large_model_logits = large_model(images)
                small_model_logits = small_model(images)
                soft_labels = torch.softmax(large_model_logits / 1, dim=1)
                loss_distillation = distillation_criterion(F.log_softmax(small_model_logits / 1, dim=1), 
                                                    soft_labels)
        
                small_seg_loss = bce_dice_loss(small_model_logits, binary_mask)
            
                total_small_loss = small_seg_loss + loss_distillation
                total_small_loss.backward()
                small_optimizer.step()


                large_model.train()
                small_model.eval()

                for param in small_model.parameters():
                    param.requires_grad = False

                for param in large_model.parameters():
                    param.requires_grad = True

                large_seg_loss = bce_dice_loss(large_model_logits, binary_mask)
                small_model_logits = small_model(images)
                large_model_logits = large_model(images)
                soft_labels = torch.softmax(large_model_logits / 1, dim=1) 
                

                with torch.no_grad():
                    small_seg_loss = bce_dice_loss(small_model_logits, binary_mask)
                
                top_k_ranking = 1
                if args.dataset == 'Fed_IXI':
                    rkd_loss = student_ranking_loss_IXI(small_model_logits, large_model_logits, top_k=top_k_ranking)
                else:
                    rkd_loss = student_ranking_loss(small_model_logits, large_model_logits, top_k=top_k_ranking)
                loss_distillation = distillation_criterion(F.log_softmax(small_model_logits / args.T, dim=1),
                                                    soft_labels)
                
                total_large_loss = large_seg_loss + loss_distillation + 0.2 * rkd_loss
                total_large_loss.backward()
                large_optimizer.step()


                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), total_large_loss.item()))
                batch_loss.append(total_large_loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return large_model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    
    def small_client_update_weights_seg(self, model, global_round): 
       

        model.to(device)
        model.train()
        epoch_loss = []

        
      
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
            
                binary_mask = convert_to_binary_mask(labels, class_index=1)
                if args.dataset == 'ChestXray_seg':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
          
                optimizer.zero_grad()
                log_probs = model(images)
             
                loss = bce_dice_loss(log_probs, binary_mask)
             
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss))
       
                batch_loss.append(loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

    def small_client_update_weights_seg_final(self, small_model, global_round, public_data):
     
        private_seg = PrivateClassifier_seg().to(device)
        public_seg = PublicClassifier_seg().to(device)
        small_model.to(device)
        small_model.train()
        epoch_loss = []


     
        if self.args.optimizer == 'sgd':
            ce_optimizer = torch.optim.SGD(list(small_model.parameters()) + list(private_seg.parameters()), lr=self.args.lr,
                                        momentum=0.5)
            kd_optimizer = torch.optim.SGD(list(small_model.parameters())+list(public_seg.parameters()), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            ce_optimizer = torch.optim.Adam(list(small_model.parameters()) + list(private_seg.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)
            kd_optimizer = torch.optim.Adam(list(small_model.parameters())+list(public_seg.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)


        for iter in range(self.args.local_ep):
            batch_loss = []
            public_data=random.sample(public_data, len(public_data))
            if len(self.trainloader) >len(public_data):
                pickled_data = public_data.repeat(len(self.trainloader)//len(public_data)+1,1,1,1)
            else:
                pickled_data = random.sample(public_data, len(self.trainloader))


            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
      
                binary_mask = convert_to_binary_mask(labels, class_index=1)
                if args.dataset == 'ChestXray_seg':
                    images = images.repeat(1,3,1,1)
                    images.to(device)
                ce_optimizer.zero_grad()
              
                feature_vector_private = small_model(images, return_feature_vect=True)
                ce_log_probs_private = private_seg(feature_vector_private)

                ce_loss = bce_dice_loss(ce_log_probs_private, binary_mask)
                ce_loss.backward()
                ce_optimizer.step()

                pub_batch=pickled_data[batch_idx]
             
                public_image=pub_batch[0].to(device)
                public_vector=pub_batch[2].to(device)
                public_mask=pub_batch[3].to(device)
              
                kd_optimizer.zero_grad()

                feature_vector_pub = small_model(public_image, return_feature_vect=True)
                kd_log_probs = public_seg(feature_vector_pub)

             
                kd_loss = distillation_criterion(F.log_softmax(kd_log_probs / 1, dim=1),
                                                    public_vector / 1) 
                kd_loss.backward()
                kd_optimizer.step()

                loss_total = ce_loss +  kd_loss
                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss_total.item()))
                batch_loss.append(loss_total.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        small_model.outconv = private_seg.outconv

        return small_model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def small_client_update_weights_seg_3d_vis(self, small_model):
        
        i = 0
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.cpu(), labels.cpu()
         
            a_index, b_index, c_index = 0, 0, 0  
            my_image = images[a_index, b_index, 1, :, :]
            my_mask = labels[a_index, 1, 1, :, :]
    
            folder_name = 'visualizations_1'  
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            plt.imshow(my_image, cmap='viridis')
            plt.axis('off') 
            image_filename = os.path.join(folder_name, 'my_image_{}.png'.format(i))
            plt.savefig(image_filename)
            plt.close()
            
            plt.imshow(my_mask, cmap='viridis')
            plt.axis('off') 
            mask_filename = os.path.join(folder_name, 'my_mask_{}.png'.format(i))
            plt.savefig(mask_filename)
            plt.close()

            i = i+1


        return 0












    def small_client_update_weights_seg_3d(self, small_model, global_round, public_data):
        
        private_seg = PrivateClassifier_seg().to(device)
        public_seg = PublicClassifier_seg().to(device)
        small_model.to(device)
        small_model.train()
        epoch_loss = []


        
        if self.args.optimizer == 'sgd':
            ce_optimizer = torch.optim.SGD(list(small_model.parameters()) + list(private_seg.parameters()), lr=self.args.lr,
                                        momentum=0.5)
            kd_optimizer = torch.optim.SGD(list(small_model.parameters())+list(public_seg.parameters()), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            ce_optimizer = torch.optim.Adam(list(small_model.parameters()) + list(private_seg.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)
            kd_optimizer = torch.optim.Adam(list(small_model.parameters())+list(public_seg.parameters()), lr=self.args.lr,
                                         weight_decay=1e-4)


        for iter in range(self.args.local_ep):
            batch_loss = []
            public_data=random.sample(public_data, len(public_data))
            pickled_data = public_data
 

            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                
                
                binary_mask = convert_to_binary_mask(labels, class_index=1)
                if args.dataset == 'ChestXray_seg':
                    images = images.repeat(1,3,1,1)
                    images.to(device)

                ce_optimizer.zero_grad()
                
                feature_vector_private = small_model(images, return_feature_vect=True)
                ce_log_probs_private = private_seg(feature_vector_private)

                ce_loss = bce_dice_loss(ce_log_probs_private, binary_mask)
                ce_loss.backward()
                ce_optimizer.step()

                pub_batch=pickled_data[batch_idx]
                
                public_image=pub_batch[0].to(device)
                public_vector=pub_batch[2].to(device)
                public_mask=pub_batch[3].to(device)
                
                kd_optimizer.zero_grad()

                feature_vector_pub = small_model(public_image, return_feature_vect=True)
                kd_log_probs = public_seg(feature_vector_pub)

             
                kd_loss = distillation_criterion(F.log_softmax(kd_log_probs / 1, dim=1),
                                                    F.log_softmax(public_vector / 1, dim=1)) 
                kd_loss.backward()
                kd_optimizer.step()

                loss_total = ce_loss +  kd_loss
                if self.args.verbose and (batch_idx % 1 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss_total.item()))
                batch_loss.append(loss_total.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        small_model.outconv = private_seg.outconv

        return small_model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def small_client_update_weights(self, model, global_round):
        
        model.to(device)
        model.train()
        epoch_loss = []

        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader)):
                images, labels = images.to(device), labels.to(device)
                
                
                if args.dataset == 'ChestXray':
                    images = images.repeat(1,3,1,1)
                    images.to(device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
    
    def seg_inference(self, model, model_dict):
        """ Returns the inference accuracy and loss.
        """
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        test_dice_scores = []
        
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)
            if args.dataset == 'ChestXray_seg':
                images = images.repeat(1,3,1,1)
                images.to(device)
            
            
            
            pred_masks_logits = model(images, return_feature_vect=False)
            pred_masks = torch.sigmoid(pred_masks_logits) 
            pred_masks = (pred_masks > 0.5).float()
            dice_score = dice_coefficient(pred_masks, labels)
            test_dice_scores.append(dice_score.item())


        avg_dice_score = np.mean(test_dice_scores)

        return avg_dice_score

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        model.to(device)
        
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(device), labels.to(device)
            if args.dataset == 'ChestXray' or args.dataset == 'ChestXray_binary':
                images = images.repeat(1,3,1,1)
                images.to(device)
            
            
            outputs = model(images, return_feature_vect=False)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss



