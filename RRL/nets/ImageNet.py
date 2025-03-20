import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
import resnet
import torchvision.models as models
from utils.config import args
class ImageNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: dimension of tags
        :param output_dim: dimensionality of the final representation
        """
        super(ImageNet, self).__init__()
        self.module_name = "image_model"
        self.resnet = resnet.resnet50(pretrained=True)
        #self.vgg_model_1000 = models.vgg19(pretrained=True)
        self.image1_model = torch.load("./{}_pth/{}_model_best.pt".format(args.dataset,args.dset))#与arg.dest同
        #print("./image_CLEF_pth/{}_model_best.pt".format(args.dset))
        # full-conv layers
        mid_num = 512
        self.bn = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        # self.fc2_2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, output_dim)

    def forward(self, x):
        feature = self.image1_model(x)
        x = self.bn(feature)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        norm = torch.norm(x, dim=1, keepdim=True)
        x = x / norm
        return x,feature

