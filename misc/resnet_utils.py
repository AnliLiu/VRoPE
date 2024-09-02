import torch
import torch.nn as nn
import torch.nn.functional as F


class myResnet(nn.Module):  #
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        # img前面增加一个维度，从3,360,640升为1，3,360,640
        # 因为网络的接收输入是一个mini-batch，image unsqueeze后第一个维度是留给batch size
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x) # 1 64 180 320
        x = self.resnet.bn1(x) # 1 64 180 320
        x = self.resnet.relu(x) # 1 64 180 320
        x = self.resnet.maxpool(x) # 1 64 90 160

        x = self.resnet.layer1(x) # 1 256 90 160
        x = self.resnet.layer2(x) # 1 512 45 80
        x = self.resnet.layer3(x) # 1 1024 23 40
        x = self.resnet.layer4(x) # 1 2048 12 20
        # x.mean(3).shape 1, 2048, 12;        x.mean(3).mean(2).shape 1, 2048;
        # torch.squeeze(input, dim=None, out=None)，不设置dim，则将input中大小为1的所有维删除
        # x.mean(3).mean(2)维数是torch.Size([1, 2048])
        # x.mean(3).mean(2).squeeze()维数是torch.Size([2048])
        fc = x.mean(3).mean(2).squeeze() # torch.Size([2048])
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)  # 14, 14, 2048
        # F.adaptive_avg_pool2d(x, [att_size, att_size]).shape   torch.Size([1, 2048, 14, 14])  x的最后两维变为14*14
        # F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().shape   torch.Size([2048, 14, 14]) squeeze()删除了维度为1
        # F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)    torch.Size([14, 14, 2048])

        return fc, att
