import torch
from torchsummary import summary
from retinanet import model

if __name__ == '__main__':
    retinanet = model.resnet50(num_classes = 1).to("cuda:0")

    # [data['img'].to(DEVICE).float(), data['annot'].to(DEVICE)]
    retinanet.training = False
    # summary(retinanet, ( 3, 384, 1280))

    # print(retinanet)
