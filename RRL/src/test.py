import numpy as np




with open('C:/Users/10189/Desktop/CoDA-main1/data_press/fake_label_data/Real_World_faketrain.txt','r') as f1:  #打开一个文件只读模式
        line1 = f1.readlines()
        print(line1)
        k1=k2=0
        for i  in range (len(line1)):
            l1 = line1[k1].strip()
            kind1 = str(l1.split('/')[4])
            k1 +=1
            #print(kind1)


