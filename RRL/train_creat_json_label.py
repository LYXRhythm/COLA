import json
import numpy as np
from utils.config import args
nosie_label = []
Real_World = []
Product = []
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
with open('./data_press/{}/{}/{}_align_train.txt'.format(args.dataset,args.dset,t),'r') as f: #打开一个文件，可写模式，，改t为对齐的一边
    line = f.readlines()

    print(len(line))
    for i  in range (len(line)):
        #line_new =line_list.replace('\n','')  #将换行符替换为空('')

        l = line[i].strip()
        label = int(l.split(' ')[1])
        #label = str(label)
        path = str(l.split(' ')[0])
        #主要是这一步 将之前列表数据转为str才能加入列表
        #line_new = path +' '+ label +'\n'
        i += 1
        #print(label)
        Real_World.append(label)
with open('./data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,s),'r') as h: #打开一个文件，可写模式
    line = h.readlines()

    print(len(line))
    for i  in range (len(line)):
        #line_new =line_list.replace('\n','')  #将换行符替换为空('')

        l = line[i].strip()
        label = int(l.split(' ')[1])
        #label = str(label)
        path = str(l.split(' ')[0])
        #主要是这一步 将之前列表数据转为str才能加入列表
        #line_new = path +' '+ label +'\n'
        i += 1
        #print(label)
        Product.append(label)
nosie_label.append(Real_World)
nosie_label.append(Product)

json.dump(nosie_label, open('./data/wiki/{}_train_label.json'.format(args.dset), "w"))