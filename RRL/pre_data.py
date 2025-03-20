import torch
import torch.utils.data as data
import os
import clustering
import time
from transforms import *
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from utils.utils import *
from utils.config import args
import h5py
import json

import torchvision.models as models
from torchvision import transforms
from PIL import Image
 


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [val.split()[0] for val in image_list]
            labels = [int(val.split()[1]) for val in image_list]
            return images, np.array(labels)
    return images

class ImageList_idx(Dataset):
    def __init__(self,
                 args,
                 image_list,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 mode='RGB',
                 domain_id=0):
        nb_classes = args.class_num
        self.imgs, self.labels = make_dataset(image_list, labels)
        self.domain_id = domain_id


        # self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index],  self.labels[index]
        #print(path,target)
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img1 = self.transform(img)
          #  img1 = np.array(img1)# list转numpy.array
          #  img1 = torch.from_numpy(img1) # array2tensor
           # print(img1)
            img2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #print(img1)
        return img1
    def __len__(self):
        return len(self.imgs)


class office_load():
    def __init__(self,args,mode):
        train_bs = args.batch_size
        if args.dataset == 'office_home':
            self.class_num = 100
        elif args.dataset == 'office31':
            self.class_num = 50
        elif args.dataset == 'image_CLEF':
            self.class_num = 24
        elif args.dataset == 'Adaptiope':
            self.class_num = 150
        elif args.dataset == 'PACS':
            self.class_num = 14
        self.mode = mode
        if args.dataset == 'office_home':
            assert args.class_num == 100
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'a':
                s = 'Art'
            elif ss == 'c':
                s = 'Clipart'
            elif ss == 'p':
                s = 'Product'
            elif ss == 'r':
                s = 'Real_World'
            else:
                raise NotImplementedError

            if tt == 'a':
                t = 'Art'
            elif tt == 'c':
                t = 'Clipart'
            elif tt == 'p':
                t = 'Product'
            elif tt == 'r':
                t = 'Real_World'
            else:
                raise NotImplementedError
            #print(s,t)
            if args.experiment == 'true':
                s_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
                s_ts_path = './data_press/{}/'.format(args.dataset) + s + '_test.txt'
                t_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#改
                t_ts_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#改
            elif args.experiment == 'xr':
                s_tr_path = '.xr/data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
                s_ts_path = '.xr/data_press/{}/'.format(args.dataset) + s + '_test.txt'
                t_tr_path = '.xr/data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#改
                t_ts_path = '.xr/data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#改
            #print(s_tr_path,s_ts_path,t_tr_path,t_ts_path)
            if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
                s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        
            if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
                t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
        elif args.dataset == 'office31':
            assert args.class_num == 50
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'a':
                s = 'amazon'
            elif ss == 'd':
                s = 'dslr'
            elif ss == 'w':
                s = 'webcam'
            else:
                raise NotImplementedError

            if tt == 'a':
                t = 'amazon'
            elif tt == 'd':
                t = 'dslr'
            elif tt == 'w':
                t = 'webcam'
            else:
                raise NotImplementedError
            #print(s,t)
            
            s_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
            s_ts_path = './data_press/{}/'.format(args.dataset) + s + '_test.txt'
            t_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#标签对齐改
            t_ts_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#标签对齐改,加入args.dset
            #print(s_tr_path)
            if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
                s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
                #print('1')
            if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
                t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
        if args.dataset == 'image_CLEF':
            assert args.class_num == 24
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'b':
                s = 'b'
            elif ss == 'c':
                s = 'c'
            elif ss == 'i':
                s = 'i'
            elif ss == 'p':
                s = 'p'
            else:
                raise NotImplementedError
            if tt == 'b':
                t = 'b'
            elif tt == 'c':
                t = 'c'
            elif tt == 'i':
                t = 'i'
            elif tt == 'p':
                t = 'p'
            else:
                raise NotImplementedError
            #print(s,t)
            
            s_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
            s_ts_path = './data_press/{}/'.format(args.dataset) + s + '_test.txt'
            t_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#改
            t_ts_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#改
            #print(s_tr_path,s_ts_path,t_tr_path,t_ts_path)
            if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
                s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        
            if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
                t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
        if args.dataset == 'Adaptiope':
            assert args.class_num == 150
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'p':
                s = 'product_images'
            elif ss == 'r':
                s = 'real_life'
            elif ss == 's':
                s = 'synthetic'
            else:
                raise NotImplementedError
            if tt == 'p':
                t = 'product_images'
            elif tt == 'r':
                t = 'real_life'
            elif tt == 's':
                t = 'synthetic'
            else:
                raise NotImplementedError
            #print(s,t)
            
            s_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
            s_ts_path = './data_press/{}/'.format(args.dataset) + s + '_test.txt'
            t_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#改
            t_ts_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#改
            #print(s_tr_path,s_ts_path,t_tr_path,t_ts_path)
            if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
                s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        
            if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
                t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
            
        if args.dataset == 'PACS':
            assert args.class_num == 14
            ss = args.dset.split('2')[0]
            tt = args.dset.split('2')[1]
            if ss == 'a':
                s = 'art_painting'
            elif ss == 'c':
                s = 'cartoon'
            elif ss == 'p':
                s = 'photo'
            elif ss == 's':
                s = 'sketch'
            else:
                raise NotImplementedError
            if tt == 'a':
                t = 'art_painting'
            elif tt == 'c':
                t = 'cartoon'
            elif tt == 'p':
                t = 'photo'
            elif tt == 's':
                t = 'sketch'
            else:
                 raise NotImplementedError
            #print(s,t)
            
            s_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset)+ s +'_best1_train.txt'
            s_ts_path = './data_press/{}/'.format(args.dataset) + s + '_test.txt'
            t_tr_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_align_train.txt'#改
            t_ts_path = './data_press/{}/{}/'.format(args.dataset,args.dset) + t + '_test1.txt'#改
            #print(s_tr_path,s_ts_path,t_tr_path,t_ts_path)
            if os.path.exists(s_tr_path) and os.path.exists(s_ts_path):
                s_tr, s_ts = open(s_tr_path).readlines(), open(s_ts_path).readlines()
        
            if os.path.exists(t_tr_path) and os.path.exists(t_ts_path):
                t_tr, t_ts = open(t_tr_path).readlines(), open(t_ts_path).readlines()
        
           
        prep_dict = {}
        prep_dict['source'] = image_train()
        prep_dict['target'] = image_target()
        prep_dict['test'] = image_test()
        train_source = ImageList_idx(args, s_tr, transform=prep_dict['source'], domain_id=0)
        test_source = ImageList_idx(args, s_ts, transform=prep_dict['test'], domain_id=0)
        eval_train_source = ImageList_idx(args, s_tr, transform=prep_dict['source'], domain_id=0)
        train_target = ImageList_idx(args, t_tr, transform=prep_dict['target'], domain_id=1)
        test_target = ImageList_idx(args, t_ts, transform=prep_dict['test'], domain_id=1)
        eval_train_target = ImageList_idx(args, t_tr, transform=prep_dict['target'], domain_id=1)

        self.r = 0

        root_dir = 'mydata/'
        if mode == 'train':
            noise_file = os.path.join('./data/wiki/{}_train_label.json'.format(args.dset))
        elif mode == 'test' or 'valid':
            noise_file = os.path.join('./data/wiki/{}_test_label.json'.format(args.dset))

        noise_label = json.load(open(noise_file, "r"))
        self.noise_label = noise_label
        if mode == 'train':
            train_data =[train_target,train_source]#任务p2r，r在前，image是r
        elif mode == 'test' or 'valid':
            train_data =[test_target,test_source]

        self.train_data = train_data
        #print(self.noiee_label)
    
    def __getitem__(self, index):
        if self.mode == 'test'or'valid'or'train':
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.noise_label[v][index] for v in range(len(self.train_data))], index
        elif self.mode == 't':
            # Get data for both domains
            data_source = [self.train_data[0][index % len(self.train_data[0])]]
            noise_label_source = [self.noise_label[0][index % len(self.train_data[0])]]
            #print(len(self.train_data[1]))

            data_target = [self.train_data[1][index % len(self.train_data[1])]]
            noise_label_target = [self.noise_label[1][index % len(self.train_data[1])]]

            # Convert lists to tensors
            data_source = torch.stack(data_source).squeeze(dim=0)
            noise_label_source = torch.tensor(noise_label_source, dtype=torch.long).squeeze(dim=0)  # Assuming labels are of type int
            data_target = torch.stack(data_target).squeeze(dim=0)
            noise_label_target = torch.tensor(noise_label_target, dtype=torch.long).squeeze(dim=0)  # Assuming labels are of type int

            return [data_target, data_source], [noise_label_target, noise_label_source], index
    def __len__(self):
        #print(len(self.train_data[0]))
        if self.mode ==  'test'or'valid'or'train':
            return len(self.train_data[0])
        elif self.mode == 't':
                larger_domain_length = len(self.train_data[0]) if len(self.train_data[0]) >= len(self.train_data[1]) else len(self.train_data[1])
        return larger_domain_length

    '''
    def __getitem__(self, index):

        # Get data for both domains
        data_source = [self.train_data[0][index % len(self.train_data[0])]]
        noise_label_source = [self.noise_label[0][index % len(self.train_data[0])]]

        data_target = [self.train_data[1][index % len(self.train_data[1])]]
        noise_label_target = [self.noise_label[1][index % len(self.train_data[1])]]

        # Convert lists to tensors
        data_source = torch.stack(data_source).squeeze(dim=0)
        noise_label_source = torch.tensor(noise_label_source, dtype=torch.long).squeeze(dim=0)  # Assuming labels are of type int
        data_target = torch.stack(data_target).squeeze(dim=0)
        noise_label_target = torch.tensor(noise_label_target, dtype=torch.long).squeeze(dim=0)  # Assuming labels are of type int

        return [data_target, data_source], [noise_label_target, noise_label_source], index
    def __len__(self):
        # 选择数据量较大的域作为整个数据集的长度
        larger_domain_length = len(self.train_data[0]) if len(self.train_data[0]) >= len(self.train_data[1]) else len(self.train_data[1])
        return larger_domain_length
    '''
        








 