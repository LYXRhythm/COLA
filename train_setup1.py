import argparse
import os, sys

import numpy as np

sys.path.append('./')

import os.path as osp
import torch

import random
from collate import SimCLRCollateFunction
from Trainer import trainer_coda
from data_clus import office_load_idx
import clustering
from network import Model
from utils.utils import *



def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(
        description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=32,
                        help="maximum epoch")
    parser.add_argument("--lr_update",
                        default=20,
                        type=int,
                        help="Number of epochs to update the learning rate.")
    parser.add_argument("--pretrain_epoch",
                        default=30,#20,
                        type=int,
                        help="warm up epochs")
    parser.add_argument("--warmup_epoch2",
                        default=10,#5,
                        type=int,
                        help="warm up epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=4,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='b2p')
    parser.add_argument('--interval_epoch', type=int, default=1)
    parser.add_argument('--lr',
                        type=float,
                        default=3e-3,
                        help="learning rate")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--class_num', type=int, default=12)#x修改为准确数值（65，12，31，123）
    # parser.add_argument('--cluster_num_list', type=list, default=[65, 130, 195, 260])
    parser.add_argument('--backbone_output', type=int, default=2048)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--pretretrained', type=str2bool,
                        default=True,
                        help='use pretrained model')
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--arch',
                        type=str,
                        default="resnet50",
                        choices=["resnet50", "resnet18"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='Office-Home')
    parser.add_argument('--file', type=str, default='target')
    parser.add_argument('--office31', action='store_true')
    parser.add_argument('--dataset', default='image_CLEF', choices=['office_home', 'office31',
                                                                     'image_CLEF',
                                                                     ])
    parser.add_argument('--nce-t', default=0.1, type=float,
                        metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--nce-m', default=0.95, type=float,
                        metavar='M', help='momentum for non-parametric updates')
    parser.add_argument('--low_dim', type=int, default=512)
    parser.add_argument('--in_domain',
                        type=str2bool,
                        default=True,
                        help='in domain learning')
    parser.add_argument('--cross_domain',
                        type=str2bool,
                        default=True,
                        help='cross domain learning')
    parser.add_argument('--kmeans_all_features',
                        type=str2bool,
                        default=True,
                        help='kmeans all features to initialize clusters centroids')

    parser.add_argument('--lambda_cross_domain', default=0.01, type=float,
                        metavar='T', help='parameter for cross domain loss')
    parser.add_argument('--cross_domain_loss', default='l1', type=str,
                        choices=['l1', 'l2'], help='type of cross domain loss')
    parser.add_argument('--cross_domain_softmax',
                        type=str2bool,
                        default=False,
                        help='use softmax after cross-domain logits')

    parser.add_argument('--verbose', action='store_true', default=True,
                        help='verbose')

    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', type=int, default=31,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--save_prefix', type=str, default='', help='prefix for saving results')
    parser.add_argument('--cluster_num', type=int, default=24)#要改
    args = parser.parse_args()

    args.cluster_num_list = [24,48,72,96]
    #offcie31[50,75,100,125],office_home[100,200,300,400].imagec[24,48,72,96].a[150.200,250,300],PACS[14,28,42,56]
    return args


if __name__ == "__main__":

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    seed_torch(SEED)
    if args.dataset == 'office_home':
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
    elif args.dataset == 'office31':
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
    elif args.dataset == 'image_CLEF':
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
        
    current_folder = ""
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    BEST_PATH = './model_best.pth.tar'
    

    print(args)
    
    dset_loaders = office_load_idx(args)



    memorySize_s = dset_loaders['source_tr'].dataset.__len__()
    memorySize_t = dset_loaders['target'].dataset.__len__()
    args.memorySize_s = memorySize_s
    args.memorySize_t = memorySize_t
    model = Model(args, memorySize_s, memorySize_t)
    model = model.cuda() 
    model.train()
  

    best_map_s2t = 0.
    map_s2t = trainer_coda.test_target(args, dset_loaders['source_te'], dset_loaders['test'], model.net)
    print("First:{:.1f}".format(map_s2t))
    
   
    if args.pretrain_epoch > 0:
        for epoch in range(0, args.pretrain_epoch):
            print("[{}/{}] Pretrain model".format(epoch + 1, args.pretrain_epoch))
            trainer_coda.train_model(args, model, dset_loaders, epoch)
            
            print('Compute MAP of Model')
            map_s2t = trainer_coda.test_target(args, dset_loaders['source_te'], dset_loaders['test'],
                                          model.net)
            
            
            if map_s2t > best_map_s2t:
                best_map_s2t = map_s2t
                torch_pth_save = './UCRP_TRAIN2/{}_pth/{}_model_best.pt'.format(args.dataset,args.dset)
                torch.save(model.net,torch_pth_save)
                ff = open('./UCRP_TRAIN2/data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,s),'w')  #打开一个文件，可写模式
                with open('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,s),'r') as f:  #打开一个文件只读模式
                    line = f.readlines()
                    for i  in range (len(line)):
                        #line_new =line_list.replace('\n','')  #将换行符替换为空('')
                        l = line[i].strip()
                        label = int(l.split(' ')[1])
                        label = str(label)
                        path = str(l.split(' ')[0])
                        #主要是这一步 将之前列表数据转为str才能加入列表
                        line_new = path +' '+ label +'\n'
                        i += 1
                       # print(line_new)
                        ff.write(line_new) #写入一个新文件中
                ff1 = open('./UCRP_TRAIN2/data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,t),'w')  #打开一个文件，可写模式
                with open('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,t),'r') as f1:  #打开一个文件只读模式
                    line = f1.readlines()
                    for i  in range (len(line)):
                        #line_new =line_list.replace('\n','')  #将换行符替换为空('')
                        l = line[i].strip()
                        label = int(l.split(' ')[1])
                        label = str(label)
                        path = str(l.split(' ')[0])
                        #主要是这一步 将之前列表数据转为str才能加入列表
                        line_new = path +' '+ label +'\n'
                        i += 1
                       # print(line_new)
                        ff1.write(line_new) #写入一个新文件中
            str_s2t = "Task        : {}, Best:{:.1f}, Last:{:.1f}".format(args.dset, best_map_s2t, map_s2t)
            
            print(str_s2t)


    map_train_data = trainer_coda.train_target(args,dset_loaders['source_train'],dset_loaders['target_train'],model.net)
    print(map_train_data)

    with open('./read_txt/map.txt','a') as f:
        k=str(best_map_s2t)
        b=str(args.dset)
        c=str(args.pretrain_epoch)
        line = '\n'+b+' ' + c +' ' + k + '\n'
        f.write(line)
