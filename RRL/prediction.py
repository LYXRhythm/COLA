import torch
from torchvision import models, transforms
from PIL import Image
from utils.config import args
import torch.nn as nn
def prediction(newmodel,input_txt_path,output_txt_path):
    input_txt_path = input_txt_path
    output_txt_path = output_txt_path  # 替换为保存结果的txt文件路径  
    # 步骤2: 读取图片地址
    with open(input_txt_path, 'r') as file:
        lines = file.readlines()

    # 步骤3: 加载图像并进行预测
    results = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])


    for line in lines:
        parts = line.strip().split(' ')
        image_path = parts[0]  # 获取图片地址部分
        #print(image_path)
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            predictions = newmodel(img)
        #print(predictions.shape)
        predicted_class = torch.argmax(predictions).item()
        results.append((image_path, predicted_class))

    # 步骤4: 保存结果
    with open(output_txt_path, 'w') as output_file:
        for result in results:
            output_file.write(f"{result[0]} {result[1]}\n")
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 步骤1: 准备模型和权重
    #model = models.resnet50(pretrained=False)  # 替换为你的模型，可以根据实际情况调整
    #  # 替换为你模型的输出类别数
    model = torch.load('./{}_pth/{}_model_best.pt'.format(args.dataset,args.dset))
    #print(model)
    # model.append = torch.load('./office31_pth/top_lay_src/{}_src_model.pt'.format(args.dset))
    new_net1 = torch.load('./{}_pth/top_lay_src/{}_src_model.pt'.format(args.dataset,args.dset))
    new_net2 = torch.load('./{}_pth/top_lay_tgt/{}_tgt_model.pt'.format(args.dataset,args.dset))
    new_model1 = nn.Sequential(
                model,
                new_net1)
    new_model2 = nn.Sequential(
                model,
                new_net2)
    #print(new_model)
    new_model1.to(device)
    new_model1.eval()
    new_model2.to(device)
    new_model2.eval()
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
    sth = './data_press/{}/{}_train.txt'.format(args.dataset,s)
    s_out = './data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,s)
    tar = './data_press/{}/{}_train.txt'.format(args.dataset,t)
    t_out = './data_press/{}/{}/{}_best1_train.txt'.format(args.dataset,args.dset,t)
    #print(s_out)
    prediction(new_model2,sth,s_out)
    prediction(new_model2,tar,t_out)
      

