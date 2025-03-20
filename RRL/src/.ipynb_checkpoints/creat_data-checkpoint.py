import random
from logging import getLogger

import cv2
from PIL import ImageFilter, Image
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import json
from utils.config import args
from numpy.testing import assert_array_almost_equal
import h5py





class creat_my_data(data.Dataset):
    def __init__(self, dataset, mode,  root_dir='mydata/', noise_file=None, pred=False, probability=[], log=''):
        self.mode = mode
        path = os.path.join(root_dir, 'a2p_MRL.h5')  # wiki_deep_doc2vec_data
     
        valid_len = 231
        global train_label
        global train_data

        h = h5py.File(path)
        if self.mode == 'test' or self.mode == 'valid':
            test_imgs_deep = h['test_img_deep'][()].astype('float32')
            test_imgs_labels = h['test_lab'][()]
            #test_imgs_labels -= np.min(test_imgs_labels)
            try:
                test_texts_idx = test_imgs_deep
            except Exception as e:
                test_texts_idx = test_imgs_deep
                test_texts_labels = tr_img_lab
                #test_texts_labels -= np.min(test_texts_labels)
                test_data = [test_imgs_deep, test_texts_idx]
                test_labels = [test_imgs_labels, test_texts_labels]

                valid_flag = True
            try:
                valid_texts_idx = h['val_img_deep'][()].astype('float32')
            except Exception as e:
                try:
                    valid_texts_idx = h['val_img_deep'][()].astype('float32')
                except Exception as e:
                    valid_flag = False
                    valid_data = [test_data[0][0: valid_len], test_data[1][0: valid_len]]
                    valid_labels = [test_labels[0][0: valid_len], test_labels[1][0: valid_len]]

                    test_data = [test_data[0][valid_len::], test_data[1][valid_len::]]
                    test_labels = [test_labels[0][valid_len::], test_labels[1][valid_len::]]
                if valid_flag:
                    valid_imgs_deep = h['val_img_deep'][()].astype('float32')
                    valid_imgs_labels = h['val_lab'][()]
                    valid_texts_labels = valid_imgs_labels
                    #valid_texts_labels -= np.min(valid_texts_labels)
                    valid_data = [valid_imgs_deep, valid_texts_idx]
                    valid_labels = [valid_imgs_labels, valid_texts_labels]

                train_data = valid_data if self.mode == 'valid' else test_data
                train_label = valid_labels if self.mode == 'valid' else test_labels
        elif self.mode == 'train':
            tr_img = h['train_img_deep'][()].astype('float32')
            print(tr_img)
            tr_img_lab = h['train_lab'][()]
            #tr_img_lab -= np.min(tr_img_lab)
            #print(tr_img)
            try:
                tr_txt = h['train_img_deep'][()].astype('float32')
            except Exception as e:
                    tr_txt = h['train_img_deep'][()].astype('float32')
            tr_txt_lab = h['train_lab'][()]
            #tr_img_lab -= np.min(tr_img_lab)
            train_data = [tr_img,tr_txt]
            train_label = [tr_img_lab,tr_txt_lab]
        else:
                raise Exception('Have no such set mode!')
        h.close()

        train_label = [la.astype('int64') for la in train_label]
        #print(train_label[0])
        noise_label = train_label
        classes = np.unique(train_label[0])
        class_num = classes.shape[0]
        #print(class_num)
        self.class_num = class_num

        self.default_train_data = train_data
        self.default_noise_label = noise_label
        self.train_data = self.default_train_data
        self.noise_label = self.default_noise_label
        if pred:
            self.prob = [np.ones_like(ll) for ll in self.default_noise_label]
        else:
            self.prob = None

    def reset(self, pred, prob, mode='labeled'):
        if pred is None:
            self.prob = None
            self.train_data = self.default_train_data
            self.noise_label = self.default_noise_label
        elif mode == 'labeled':
            inx = np.stack(pred).sum(0) > 0.5
            self.train_data = [dd[inx] for dd in self.default_train_data]
            self.noise_label = [dd[inx] for dd in self.default_noise_label]
            probs = np.stack(prob)[:, inx]
            prob_inx = probs.argmax(0)
            labels = np.stack(self.noise_label)[prob_inx, np.arange(probs.shape[1])]
            prob = probs[prob_inx, np.arange(probs.shape[1])]
            self.noise_label = [labels for _ in range(len(self.default_noise_label))]
            self.prob = [prob, prob]
        elif mode == 'unlabeled':
            inx = np.stack(pred).sum(0) <= 0.5
            self.train_data = [dd[inx] for dd in self.default_train_data]
            self.noise_label = [dd[inx] for dd in self.default_noise_label]
            self.prob = [dd[inx] for dd in prob]
        else:
            self.train_data = self.default_train_data
            # inx = (np.stack(pred).sum(0) <= 0.5).float()
            inx = [(p <= 0.5).astype('float32') for p in pred]
            self.noise_label = [self.default_noise_label[i] * (1. - inx[i]) - inx[i] for i in range(len(self.default_noise_label))]
            self.prob = prob

    def __getitem__(self, index):
        if self.prob is None:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.noise_label[v][index] for v in range(len(self.train_data))], index
        else:
            return [self.train_data[v][index] for v in range(len(self.train_data))], [self.noise_label[v][index] for v in range(len(self.train_data))], [self.prob[v][index] for v in range(len(self.prob))], index

    def __len__(self):
        #print(len(self.train_data[0]))
        return len(self.train_data[0])