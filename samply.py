import numpy as np
from torchvision import datasets, transforms
def chestxray_seg_seprate(train_dataset,test_dataset, num_users):
    ration_list = [0.45,0.45,0.1]
    num_items = [int(len(train_dataset) * r) for r in ration_list]  
    dict_users, all_idxs = {}, [i for i in range(len(train_dataset))]
    for i in range(num_users):
        dict_users[i] = []
        dict_users[i].append(list(set(np.random.choice(all_idxs, num_items[i],
                                                       replace=False))))
        all_idxs = list(set(all_idxs) - set(dict_users[i][0]))
    num_items = [int(len(test_dataset) * r) for r in ration_list]
    all_idxs = [i for i in range(len(test_dataset))]
    for i in range(num_users):
        dict_users[i].append(list(set(np.random.choice(all_idxs, num_items[i],
                                                       replace=False))))
        all_idxs = list(set(all_idxs) - set(dict_users[i][1]))

    return dict_users

def chestxray_seprate(train_dataset,test_dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    ration_list=[0.6,0.2,0.08,0.06,0.04,0.02]
    num_items = [int(len(train_dataset)*r) for r in ration_list ]
    dict_users, all_idxs = {}, [i for i in range(len(train_dataset))]
    for i in range(num_users):
        dict_users[i]=[]
        dict_users[i].append(list(set(np.random.choice(all_idxs, num_items[i],
                                             replace=False))))
        all_idxs = list(set(all_idxs) - set(dict_users[i][0]))
    num_items = [int(len(test_dataset)*r) for r in ration_list ]
    all_idxs = [i for i in range(len(test_dataset))]
    for i in range(num_users):
        dict_users[i].append(list(set(np.random.choice(all_idxs, num_items[i],
                                             replace=False))))
        all_idxs = list(set(all_idxs) - set(dict_users[i][1]))

    return dict_users
