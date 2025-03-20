import numpy as np


ff1 = open('../data_press/fake_label_data/Clipart_test1.txt','w')  #打开一个文件，可写模式
ff2 = open('../data_press/fake_label_data/Art_test1.txt','w')  #打开一个文件，可写模式

with open('../data_press/fake_label_data/Clipart_test.txt','r') as f1:  #打开一个文件只读模式
    with open('../data_press/fake_label_data/Art_test.txt','r') as f2:  #打开一个文件只读模式
        line1 = f1.readlines()
        line2 = f2.readlines()
        z=1
        k1=k2=0
        len1=len(line1)
        len2=len(line2)
        #print(line1)
        while z==1:
            
            if k1>len1-1:
                break
            if k2>len2-1:
                break
            #print(l1)
            l1 = line1[k1].strip()
            l2 = line2[k2].strip()
            kind1 = str(l1.split('/')[4])
           # print(kind1)
            kind2 = str(l2.split('/')[4])
            #print(k1)
            #print(k2)
            if k1 != 0:
                l11 = line1[k1-1].strip()
                l22 = line2[k2-1].strip()
                kind11 = str(l11.split('/')[4])
                kind22 = str(l22.split('/')[4])

            if kind1 == kind2 :
                k1 += 1
                k2 += 1
                l1=str(l1)
                l2=str(l2)
                new1 = l1 + '\n'
                new2 = l2 + '\n'
                ff1.write(new1) #写入一个新文件中
                ff2.write(new2)
            elif kind1 != kind2 :
                if kind1 != kind11:
                    k2 += 1
                elif kind2 != kind22:
                    k1 += 1