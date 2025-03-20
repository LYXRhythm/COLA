import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 加载特征网络
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # 加载已训练好的.pth文件
        #self.feature_network = torch.load('./dest_pth/a2r.pth')
        self.feature_network = torch.load('./office_home_pth/c2p_model_best.pt')
        self.feature_network.eval()

    def forward(self, x):
        #features, _ = self.feature_network(x)  # 假设特征网络返回特征和另一个值
        features = self.feature_network(x)  # 假设特征网络返回特征和另一个值
        return features


# 图像预处理
def image_loader(image_name):
    image = Image.open(image_name)
    loader = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = loader(image).float()
    image = image.unsqueeze(0)  # 添加 batch 维度
    return image

# 计算相似度
def compute_similarity(feature1, feature2):
    cos = nn.CosineSimilarity(dim=1)
    similarity = cos(feature1, feature2)
    return similarity.item()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载特征提取网络
feature_extractor = FeatureExtractor()

feature_extractor.to(device)

# 加载目标域图片文件夹
target_folder = './datasets/OfficeHomeDataset_10072016/Product'

# 加载源域图片
source_image_path = './datasets/OfficeHomeDataset_10072016/Clipart/Drill/00005.jpg'
source_image = image_loader(source_image_path)
source_image = source_image.to(device)
source_feature = feature_extractor(source_image)
#print(source_feature)
#source_feature = torch.tensor(source_feature)
# 遍历目标域图片文件夹
similar_images = []
for root, dirs, files in os.walk(target_folder):
    for file in files:
        if file.endswith(".jpg"):
            target_image_path = os.path.join(root, file)
            target_image = image_loader(target_image_path)
            #print(target_image_path)
            target_image = target_image.to(device)
            target_feature = feature_extractor(target_image)
            #target_feature = torch.tensor(target_feature)
            similarity = compute_similarity(source_feature, target_feature)
            similar_images.append((target_image_path, similarity))

# 按相似度排序
similar_images.sort(key=lambda x: x[1], reverse=True)
# 保存最相近的10张图片
output_folder = 'similar_images'
os.makedirs(output_folder, exist_ok=True)
for i in range(min(10, len(similar_images))):
    image_path, similarity = similar_images[i]
    print(image_path)
    image_name = os.path.basename(image_path)
    new_image_path = os.path.join(output_folder, f"{similarity:.4f}_{image_name}")
    # 保存图片到指定文件夹
    with open(image_path, 'rb') as f:
        image_data = f.read()
        with open(new_image_path, 'wb') as nf:
            nf.write(image_data)

