import sys
sys.path.append("..")
import torch
import torch.nn as nn
from utils.utils import *
from utils.utils import recompute_memory, get_features,add_fake_labels
import h5py
from clustering import compute_variance, torch_kmeans
import os
from utils.utils import GECELoss
from utils.utils import GCELoss
def train_model(args, model, dset_loaders, epoch):
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
    with open('./UCRP_TRAIN2/data_press/{}/{}_train.txt'.format(args.dataset,s), 'r') as file:
        all_lines = file.readlines()
        
    with open('./UCRP_TRAIN2/data_press/{}/{}_train.txt'.format(args.dataset,t), 'r') as file:
        all_lines1 = file.readlines()
        print(len(all_lines1))
    source_loader = dset_loaders["source_tr"]
    target_loader_unl = dset_loaders["target"]

    criterion = GECELoss(q=0.001).cuda()

    if model.lemniscate_s.memory_first:
        recompute_memory(0, model, model.lemniscate_s, source_loader)

    if model.lemniscate_t.memory_first:
        recompute_memory(0, model, model.lemniscate_t, target_loader_unl)

    if epoch == 0:
        
        features = torch.cat((model.lemniscate_s.memory, model.lemniscate_t.memory), 0)

        for i in range(len(args.cluster_num_list)):
            cluster_num = args.cluster_num_list[i]
            init_centroids = None

            if args.kmeans_all_features:
                cluster_labels, cluster_centroids, cluster_phi= torch_kmeans([cluster_num], features, seed=0)
                init_centroids = cluster_centroids[0]

            cluster_labels_src, cluster_centroids_src, cluster_phi_src = torch_kmeans([cluster_num],
                                                                                      model.lemniscate_s.memory,
                                                                                      init_centroids=init_centroids,
                                                                                      seed=0)
            cluster_labels_tgt, cluster_centroids_tgt, cluster_phi_tgt = torch_kmeans([cluster_num],
                                                                                      model.lemniscate_t.memory,
                                                                                      init_centroids=init_centroids,
                                                                                      seed=0)
            weight_src = cluster_centroids_src[0]
            weight_tgt = cluster_centroids_tgt[0]
            with torch.no_grad():
                model.top_layer_src[i].weight.copy_(weight_src)
                model.top_layer_tgt[i].weight.copy_(weight_tgt)

    model.train()

    for batch_idx, (inputs_src1, _, domainIDs_src, indexes_src,path_src) in enumerate(source_loader):
        try:
            inputs_tar1, _, domainIDs_tar, indexes_tar,path_tar = next(target_loader_unl_iter)
        except:
            target_loader_unl_iter = iter(target_loader_unl)
            inputs_tar1, _, domainIDs_tar, indexes_tar,path_tar = next(target_loader_unl_iter)

        inputs_src1,  domainIDs_src, indexes_src = inputs_src1.cuda(),  domainIDs_src.cuda(), indexes_src.cuda()
        inputs_tar1,  domainIDs_tar, indexes_tar = inputs_tar1.cuda(), domainIDs_tar.cuda(), indexes_tar.cuda()

        model.optimizer.zero_grad()
        model.optimizer_tl_src.zero_grad()
        model.optimizer_tl_tgt.zero_grad()


        logits_src1, features_src1 = model(inputs_src1, head='src')
        logits_tar1, features_tar1 = model(inputs_tar1, head='tgt')


        tau = 0.01
        
        loss_iss1 = 0
        loss_iss2 = 0
        loss_cca1 = 0
        loss_cca2 = 0
        for i in range(len(args.cluster_num_list)):
            if args.in_domain:
                logits_mem_src1 = model.top_layer_tgt[i](model.lemniscate_s.memory[indexes_src])
                a = (logits_mem_src1 / tau).softmax(1).detach().shape[0]
          
                fake = [0]*a
                fake_label = [0]*a
                for k in range(a):
                    fake[k],fake_label[k] = torch.max((logits_mem_src1 / tau).softmax(1).detach()[k],0)

                fake_numpy = np.array([tensor.cpu().detach().numpy() for tensor in fake])
                fake_label_numpy = np.array([tensor.cpu().detach().numpy() for tensor in fake_label])
                loss_iss1 += criterion(logits_src1[i], torch.tensor(fake_label_numpy).cuda())
                #print(torch.tensor(fake_label_numpy).cuda())
                if os.path.exists('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,s)):
                    with open('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,s), 'r') as file1:
                        line = file1.readlines()
                else:
                    line = all_lines
                file_path = './UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,s)
                if i == 0:
                    for z in range(a):
                        save_array_element_to_txt(line,path_src,fake_label_numpy,z,file_path,indexes_src[z])

                        
                logits_mem_tar1 = model.top_layer_tgt[i](model.lemniscate_t.memory[indexes_tar])

                b = (logits_mem_tar1 / tau).softmax(1).detach().shape[0]
              
                fake1 = [0]*b
                fake_label1 = [0]*b
                for k1 in range(b):
                    fake1[k1],fake_label1[k1] = torch.max((logits_mem_tar1 / tau).softmax(1).detach()[k1],0)
                fake_numpy1 = np.array([tensor.cpu().detach().numpy() for tensor in fake1])
                fake_label_numpy1 = np.array([tensor.cpu().detach().numpy() for tensor in fake_label1])
                loss_iss2 += criterion(logits_tar1[i], torch.tensor(fake_label_numpy1).cuda())      
                if os.path.exists('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,t)):
                    with open('./UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,t),'r') as file2:
                        line1 = file2.readlines()
                else:
                    line1 = all_lines1
                file_path1 = './UCRP_TRAIN2/data_press/{}/{}/{}_cluster_train.txt'.format(args.dataset,args.dset,t)
                if i == 0:
                    for z1 in range(b):
                        save_array_element_to_txt(line1,path_tar,fake_label_numpy1,z1,file_path1,indexes_tar[z1])
                        
                        
            if args.cross_domain:

                outputs_src_cls_tgt = model.top_layer_src[i](features_tar1)
                outputs_tgt_cls_tgt = model.top_layer_tgt[i](features_tar1)
                outputs_tgt_cls_src = model.top_layer_tgt[i](features_src1)
                outputs_src_cls_src = model.top_layer_src[i](features_src1)

                if args.cross_domain_softmax:
                    outputs_src_cls_tgt = outputs_src_cls_tgt.softmax(1)
                    outputs_tgt_cls_tgt = outputs_tgt_cls_tgt.softmax(1)
                    outputs_tgt_cls_src = outputs_tgt_cls_src.softmax(1)
                    outputs_src_cls_src = outputs_src_cls_src.softmax(1)
                if args.cross_domain_loss == 'l1':
                    loss_cca1 += args.lambda_cross_domain*(
                        (outputs_src_cls_tgt - outputs_tgt_cls_tgt).abs().sum(1).mean())
                    loss_cca2 += args.lambda_cross_domain * (
                        (outputs_src_cls_src - outputs_tgt_cls_src).abs().sum(1).mean())
                    
                elif args.cross_domain_loss == 'l2':
                    loss_cca1 += args.lambda_cross_domain * ((
                            (outputs_src_cls_tgt - outputs_tgt_cls_tgt) ** 2).sum(1).mean())
                    loss_cca2 += args.lambda_cross_domain * ((
                            (outputs_src_cls_src - outputs_tgt_cls_src) ** 2).sum(1).mean())
                    
                
            if (not args.in_domain) and (not args.cross_domain):
                raise InterruptedError

        loss_iss1 /= len(args.cluster_num_list)
        loss_iss2 /= len(args.cluster_num_list)
        loss_cca1 /= len(args.cluster_num_list)
        loss_cca2 /= len(args.cluster_num_list)
    

        loss_cdm = (loss_iss1 + loss_iss2) + (loss_cca1 + loss_cca2)
        #loss_cdm = (loss_iss1 + loss_iss2)
        #loss_cdm = (loss_cca1 + loss_cca2)
        loss_cdm.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()
        model.optimizer_tl_src.step()
        model.optimizer_tl_src.zero_grad()
        model.optimizer_tl_tgt.step()
        model.optimizer_tl_tgt.zero_grad()
        model.lemniscate_s.update_weight(features_src1.detach(), indexes_src)
        model.lemniscate_t.update_weight(features_tar1.detach(), indexes_tar)

        if batch_idx % 50 == 0:
            print(f"Step [{batch_idx}/{len(source_loader)}] loss_iss1: {loss_iss1} loss_iss2: {loss_iss2} loss_cca1: {loss_cca1} loss_cca2: {loss_cca2}")
    model.lemniscate_s.memory_first = False
    model.lemniscate_t.memory_first = False

def save_array_element_to_txt(line,path,array, element_index, file_path, line_number):
    # 将数组的第8个元素转换为字符串
    data_label = str(array[element_index])
    path1 = str(path[element_index])
    # 创建一个新的列表，将字符串添加到其中
    lines = [path1+ ' ' +data_label + '\n']

    # 将新内容写入到txt文件中
    all_lines = line
    all_lines[line_number] = lines[0]
    #print(all_lines[line_number])
    with open(file_path, 'w') as file:
        file.writelines(all_lines)    
    
    
import numpy as np

def test_target(args, src_loader, tgt_loader, netF, path_for_save_features=None):
    netF.eval()

    print('Prepare Gallery Features.....')
    features_gallery, gt_labels_gallery = get_features(src_loader, netF)

    print('Prepare Query Features of Target Domain.....')
    features_query, gt_labels_query = get_features(tgt_loader, netF)

    if path_for_save_features:
        with h5py.File(path_for_save_features, 'w') as hf:
            hf.create_dataset('features_gallery', data=features_gallery)
            hf.create_dataset('gt_labels_gallery', data=gt_labels_gallery)
            hf.create_dataset('features_query', data=features_query)
            hf.create_dataset('gt_labels_query', data=gt_labels_query)

    fake_labelling = np.zeros((features_query.shape[0],), dtype = int)   

    map_t,fake_labelling = cal_map_sda(features_query, gt_labels_query,
                        features_gallery, gt_labels_gallery)
   # print(fake_labelling)
    #if args.pretrain_epoch > 5 :
      #  add_fake_test_labels(fake_labelling,args)

    return map_t * 100

def train_target(args, src_loader, tgt_loader, netF, path_for_save_features=None):
    netF.eval()

    print('Prepare Gallery Features.....')
    features_gallery, gt_labels_gallery = get_features(src_loader, netF)

    print('Prepare Query Features of Target Domain.....')
    features_query, gt_labels_query = get_features(tgt_loader, netF)

    if path_for_save_features:
        with h5py.File(path_for_save_features, 'w') as hf:
            hf.create_dataset('features_gallery', data=features_gallery)
            hf.create_dataset('gt_labels_gallery', data=gt_labels_gallery)
            hf.create_dataset('features_query', data=features_query)
            hf.create_dataset('gt_labels_query', data=gt_labels_query)

    fake_labelling = np.zeros((features_query.shape[0],), dtype = int)   

    map_t,fake_labelling = cal_map_sda(features_query, gt_labels_query,
                        features_gallery, gt_labels_gallery)
   # print(fake_labelling)
   
    if args.pretrain_epoch > 5 :
        add_fake_labels(fake_labelling,args)

    return map_t * 100



# def test_save(args, src_loader, tgt_loader, netF, path_for_save_features=None):
#     netF.eval()

