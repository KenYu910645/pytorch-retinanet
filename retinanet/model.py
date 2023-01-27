import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import pickle

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        # C3_size = torch.Size([1, 512, 48, 160])
        # C4_size = torch.Size([1, 1024, 24, 80])
        # C5_size = torch.Size([1, 2048, 12, 40])

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs # (1/8, 1/16, 1/32)

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x) # 1/32

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x) # 1/16

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x) # 1/8

        P6_x = self.P6(C5) # 1/64

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x) # 1/128

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)

class My_RegressionModel(nn.Module):
    # NOte that RetinaNet use share weight to predict multi-scale
    def __init__(self, num_features_in, crop_range, num_anchors=9, feature_size=256):
        super(My_RegressionModel, self).__init__()
        self.crop_range = crop_range
        
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # crop the feature map according to the argument
        # print(f"out.shape = {out.shape}") [B, C, H, W]
        # out.shape = torch.Size([8, 24, 48, 160])
        # out.shape = torch.Size([8, 40, 24, 80])
        # out.shape = torch.Size([8, 112, 12, 40])
        # out.shape = torch.Size([8, 512, 6, 20])
        # out.shape = torch.Size([8, 1792, 3, 10])
        out = out[:, :, self.crop_range[0]:self.crop_range[1], :]
        
        # out is B x C x H x W, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1) # [B, H, W, C]
        
        # out.shpae = torch.Size([8, 5, 160, 24]) # 6 *4 
        # out.shpae = torch.Size([8, 4, 80, 40])  # 10*4
        # out.shpae = torch.Size([8, 3, 40, 112]) # 28*4
        # out.shpae = torch.Size([8, 2, 20, 512])
        # out.shpae = torch.Size([8, 1, 10, 1792])
        # anchor_count = 6 at level 3
        # anchor_count = 10 at level 4
        # anchor_count = 28 at level 5
        # anchor_count = 128 at level 6
        # anchor_count = 448 at level 7
        # print(f"out.contiguous().view(out.shape[0], -1, 4) = {out.contiguous().view(out.shape[0], -1, 4).shape}")
        # torch.Size([8, 4800, 4])
        return out.contiguous().view(out.shape[0], -1, 4)

class My_ClassificationModel(nn.Module):
    def __init__(self, num_features_in, crop_range, num_anchors=9, num_classes=1, prior=0.01, feature_size=256):
        super(My_ClassificationModel, self).__init__()
        self.crop_range = crop_range

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # crop the feature map according to the argument
        out = out[:, :, self.crop_range[0]:self.crop_range[1], :]
        
        # print(f"out = {out.shape}")
        # out = torch.Size([8, 6, 5, 160])
        # out = torch.Size([8, 10, 4, 80])
        # out = torch.Size([8, 28, 3, 40])
        # out = torch.Size([8, 128, 2, 20])
        # out = torch.Size([8, 448, 1, 10])
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)
 
        batch_size, width, height, channels = out1.shape
        
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        # print(f"out1.shape = {out1.shape}")
        # print(f"out2.shape = {out2.shape}")
        # out1.shape = torch.Size([8, 1, 10, 448])
        # out2.shape = torch.Size([8, 1, 10, 448, 1])
        
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, mode, device):
        self.inplanes = 64
        self.mode = mode
        self.device = device

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)        

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest')
        
        if mode == 'original':
            self.regressionModel = RegressionModel(256)
            self.classificationModel = ClassificationModel(256, num_classes=num_classes)
            
            # Init weight
            prior = 0.01
            self.classificationModel.output.weight.data.fill_(0)
            self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
            self.regressionModel.output.weight.data.fill_(0)
            self.regressionModel.output.bias.data.fill_(0)
        
        elif mode == "crop_feature_map":
            # anchor_count = 6   at level 3
            # anchor_count = 10  at level 4
            # anchor_count = 28  at level 5
            # anchor_count = 128 at level 6
            # anchor_count = 448 at level 7

            # my regression model
            self.reg_8   = My_RegressionModel(256, (21, 26), num_anchors=6)   # [21, 22, 23, 24, 25]
            self.reg_16  = My_RegressionModel(256, (10, 14), num_anchors=10)  # [10, 11, 12, 13]
            self.reg_32  = My_RegressionModel(256, (5, 8),   num_anchors=28)  # [5, 6, 7]
            self.reg_64  = My_RegressionModel(256, (3, 5),   num_anchors=128) # [3, 4]
            self.reg_128 = My_RegressionModel(256, (2, 3),   num_anchors=448) # [2]
            self.crop_reg_net = [self.reg_8, self.reg_16, self.reg_32, self.reg_64, self.reg_128]

            # My classification model
            self.cls_8   = My_ClassificationModel(256, (21, 26), num_anchors=6)   # [21, 22, 23, 24, 25]
            self.cls_16  = My_ClassificationModel(256, (10, 14), num_anchors=10)  # [10, 11, 12, 13]
            self.cls_32  = My_ClassificationModel(256, (5, 8),   num_anchors=28)  # [5, 6, 7]
            self.cls_64  = My_ClassificationModel(256, (3, 5),   num_anchors=128) # [3, 4]
            self.cls_128 = My_ClassificationModel(256, (2, 3),   num_anchors=448) # [2]
            self.crop_cls_net = [self.cls_8, self.cls_16, self.cls_32, self.cls_64, self.cls_128]
            
            # Init weight
            prior = 0.01
            [net.output.weight.data.fill_(0) for net in self.crop_cls_net]
            [net.output.bias.data.fill_(-math.log((1.0 - prior) / prior)) for net in self.crop_cls_net]
            [net.output.weight.data.fill_(0) for net in self.crop_reg_net]
            [net.output.bias.data.fill_(0) for net in self.crop_reg_net]
        
        elif mode == "crop_feature_map_fuse_16":
            # anchor_count = 6 at level 3
            # anchor_count = 10 at level 4
            # anchor_count = 7 at level 5
            # anchor_count = 8 at level 6
            # anchor_count = 7 at level 7
            
            # my regression model
            self.reg_8   = My_RegressionModel(256, (21, 26), num_anchors=6)  # [21, 22, 23, 24, 25]
            self.reg_16  = My_RegressionModel(256, (10, 14), num_anchors=10) # [10, 11, 12, 13]
            self.reg_32  = My_RegressionModel(512, (10, 16), num_anchors=7, feature_size=512)  # [10, 11, 12, 13 ,14 ,15]
            self.reg_64  = My_RegressionModel(512, (12, 20), num_anchors=8, feature_size=512)  # [12, 13, 14 ,15 ,16 ,17, 18, 19]
            self.reg_128 = My_RegressionModel(512, (16, 24), num_anchors=7, feature_size=512)  # [16, 17, 18, 19 ,20 ,21, 22, 23]
            self.crop_reg_net = [self.reg_8, self.reg_16, self.reg_32, self.reg_64, self.reg_128]

            # My classification model
            self.cls_8   = My_ClassificationModel(256, (21, 26), num_anchors=6)  # [21, 22, 23, 24, 25]
            self.cls_16  = My_ClassificationModel(256, (10, 14), num_anchors=10) # [10, 11, 12, 13]
            self.cls_32  = My_ClassificationModel(512, (10, 16), num_anchors=7, feature_size=512)  # [10, 11, 12, 13 ,14 ,15]
            self.cls_64  = My_ClassificationModel(512, (12, 20), num_anchors=8, feature_size=512)  # [12, 13, 14 ,15 ,16 ,17, 18, 19]
            self.cls_128 = My_ClassificationModel(512, (16, 24), num_anchors=7, feature_size=512)  # [16, 17, 18, 19 ,20 ,21, 22, 23]
            self.crop_cls_net = [self.cls_8, self.cls_16, self.cls_32, self.cls_64, self.cls_128]
            
            # Init weight
            prior = 0.01
            [net.output.weight.data.fill_(0) for net in self.crop_cls_net]
            [net.output.bias.data.fill_(-math.log((1.0 - prior) / prior)) for net in self.crop_cls_net]
            [net.output.weight.data.fill_(0) for net in self.crop_reg_net]
            [net.output.bias.data.fill_(0) for net in self.crop_reg_net]
        
        
        elif mode == "all_feature_map":
            # My regression model
            self.reg_8   = My_RegressionModel(256, (0, 48), num_anchors=6)
            self.reg_16  = My_RegressionModel(256, (0, 24), num_anchors=10)
            self.reg_32  = My_RegressionModel(256, (0, 12), num_anchors=28)
            self.reg_64  = My_RegressionModel(256, (0, 6),  num_anchors=128)
            self.reg_128 = My_RegressionModel(256, (0, 3),  num_anchors=448)
            self.crop_reg_net = [self.reg_8, self.reg_16, self.reg_32, self.reg_64, self.reg_128]

            # My classification model
            self.cls_8   = My_ClassificationModel(256, (0, 48), num_anchors=6)
            self.cls_16  = My_ClassificationModel(256, (0, 24), num_anchors=10)
            self.cls_32  = My_ClassificationModel(256, (0, 12), num_anchors=28)
            self.cls_64  = My_ClassificationModel(256, (0, 6),  num_anchors=128)
            self.cls_128 = My_ClassificationModel(256, (0, 3),  num_anchors=448)
            self.crop_cls_net = [self.cls_8, self.cls_16, self.cls_32, self.cls_64, self.cls_128]

            # Init weight
            prior = 0.01
            [net.output.weight.data.fill_(0) for net in self.crop_cls_net]
            [net.output.bias.data.fill_(-math.log((1.0 - prior) / prior)) for net in self.crop_cls_net]
            [net.output.weight.data.fill_(0) for net in self.crop_reg_net]
            [net.output.bias.data.fill_(0) for net in self.crop_reg_net]
            
        if mode == 'original':
            self.anchors = Anchors()
        elif mode == 'crop_feature_map':
            with open("/home/lab530/KenYu/ml_toolkit/anchor_generation/pkl/anchors_fpn_2D.pkl", "rb") as f:
                anchors = pickle.load(f).to(device)
                self.anchors = anchors.view(1, anchors.shape[0], anchors.shape[1])
                # print(anchors.shape) # torch.Size([1, 20960, 4])
        elif mode == 'all_feature_map':
            with open("/home/lab530/KenYu/ml_toolkit/anchor_generation/pkl/anchors_fpn_full_feature_2D.pkl", "rb") as f:
                anchors = pickle.load(f).to(device)
                self.anchors = anchors.view(1, anchors.shape[0], anchors.shape[1])
                # print(f"anchors.shape = {anchors.shape}")
        elif mode == 'crop_feature_map_fuse_16':
            with open("/home/lab530/KenYu/ml_toolkit/anchor_generation/pkl/anchors_fpn_fuse_16_2D.pkl", "rb") as f:
                anchors = pickle.load(f).to(device)
                self.anchors = anchors.view(1, anchors.shape[0], anchors.shape[1])
                print(anchors.shape) 

        self.regressBoxes = BBoxTransform(device = device)

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        # print(f"img_batch = {img_batch.shape}") # torch.Size([1, 3, 384, 1280])
        x = self.conv1(img_batch) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 1/4
        
        # print(f"x = {x.shape}") # x = torch.Size([1, 64, 96, 320])

        x1 = self.layer1(x) # 1/4
        x2 = self.layer2(x1) # 1/8
        x3 = self.layer3(x2) # 1/16
        x4 = self.layer4(x3) # 1/32
        # print(f"x1 = {x1.shape}") # x1 = torch.Size([1, 256, 96, 320])
        # print(f"x2 = {x2.shape}") # x2 = torch.Size([1, 512, 48, 160])
        # print(f"x3 = {x3.shape}") # x3 = torch.Size([1, 1024, 24, 80])
        # print(f"x4 = {x4.shape}") # x4 = torch.Size([1, 2048, 12, 40])

        features = self.fpn([x2, x3, x4])
        # print(f"features[0] = {features[0].shape}") # torch.Size([1, 256, 48, 160]) 1/8
        # print(f"features[1] = {features[1].shape}") # torch.Size([1, 256, 24, 80]) 1/16
        # print(f"features[2] = {features[2].shape}") # torch.Size([1, 256, 12, 40]) 1/32
        # print(f"features[3] = {features[3].shape}") # torch.Size([1, 256, 6, 20]) 1/64
        # print(f"features[4] = {features[4].shape}") # torch.Size([1, 256, 3, 10]) 1/128


        ############################################
        ### Regression and Classification Branch ###
        ############################################
        if self.mode == "original":
            regression     = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            # print(f"classification = {classification.shape}") # torch.Size([1, 92070, 1])
            # print(f"regression = {regression.shape}") # torch.Size([1, 92070, 4])
            # ( 48*160 + 24*80 + 12*40 + 6*20 + 3*10 ) * 9 = 92070

        elif self.mode == "crop_feature_map" or self.mode == "all_feature_map":
            regression     = torch.cat([reg_net(features[i]) for i, reg_net in enumerate(self.crop_reg_net)], dim=1)
            classification = torch.cat([cls_net(features[i]) for i, cls_net in enumerate(self.crop_cls_net)], dim=1)
            # print(f"classification = {classification.shape}") # torch.Size([1,  20960, 80])
            # print(f"regression = {regression.shape}") # torch.Size([1, 20960, 4])
        
        elif self.mode == "crop_feature_map_fuse_16":
            features[2] = torch.cat( [self.upsample2(features[2]), features[1]], dim=1)
            features[3] = torch.cat( [self.upsample4(features[3]), features[1]], dim=1)
            features[4] = torch.cat( [self.upsample8(features[4]), features[1]], dim=1)
            # 
            regression     = torch.cat([reg_net(features[i]) for i, reg_net in enumerate(self.crop_reg_net)], dim=1)
            classification = torch.cat([cls_net(features[i]) for i, cls_net in enumerate(self.crop_cls_net)], dim=1)
            # print(f"classification = {classification.shape}") # torch.Size([8, 20960, 1])
            # print(f"regression = {regression.shape}") # torch.Size([8, 20960, 4])

        if self.mode == "original":
            anchors = self.anchors(img_batch)
        elif self.mode in ["crop_feature_map", "all_feature_map", "crop_feature_map_fuse_16"]:
            anchors = self.anchors
            
        # print(f"anchors.shape = {anchors.shape}") # torch.Size([1, 92070, 4])

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch) #This might be a problem for 3D detector

            finalResult = [[], [], []]

            finalScores = torch.Tensor([]).to(self.device)
            finalAnchorBoxesIndexes = torch.Tensor([]).long().to(self.device)
            finalAnchorBoxesCoordinates = torch.Tensor([]).to(self.device)

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.to(self.device)

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, mode = "original", device = "cuda:0" ,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], mode, device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
