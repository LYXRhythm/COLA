import os
import random
from utils.config import args
import numpy as np

def get_labels(pth):
    with open('./data_press/{}'.format(pth),'r') as f1:  #打开一个文件只读模式
        line1 = f1.readlines()
        len1 = len(line1)
     #print(len1)
        labels = [0]*len1
        for i in range(len1-1):
            l11 = line1[i].strip()
            labels[i] = l11.split(' ')[1]
    return labels,len1

if __name__ == "__main__":
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
    elif args.dataset == 'Adaptiope':
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
    elif args.dataset == 'PACS':
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
    output_path = './data_press/{}/{}/{}_align_train.txt'.format(args.dataset,args.dset,t)
    if not os.path.exists(output_path):
        open(output_path, 'w').close()       
    flabels,flen = get_labels(pth='{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,s))
    #print(flabels,flen)
    slabels,slen = get_labels(pth='{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,t))
    with open(output_path, 'a', encoding='utf-8') as clear:
        clear.truncate(0)
    for i in range(flen):
        slabels_list = []
        for z in range(slen):
            if slabels[z] == flabels[i]:
                slabels_list.append(z)
        if slabels_list:
            random_number = random.choice(slabels_list)
        else:
            random_number = random.randint(0, slen)
        with open('./data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,t), 'r', encoding='utf-8') as file2:
            line1 = file2.readlines()
            random_sample = line1[random_number]
            with open(output_path, 'a', encoding='utf-8') as output_file:
                output_file.write(random_sample)
