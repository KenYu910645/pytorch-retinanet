import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import KittiResizer, Normalizer, KittiDataset
from retinanet import kitti_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
CATEGORY = ['Car']

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='kitti')
    parser.add_argument('--kitti_path', help='Path to KITTI directory', default='/home/lab530/KenYu/kitti/')
    parser.add_argument('--split_path', help='Path to KITTI directory', default='/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=30) #  default=100

    # parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str, default="/home/lab530/KenYu/pytorch-retinanet/kitti_retinanet_8.pt")
    # parser.add_argument('--images_path',help='Path to images directory',type=str)
    # parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    
    
    parser = parser.parse_args(args)

    # Load dataset
    dataset_val = KittiDataset(parser.kitti_path, split_path=f'{parser.split_path}val.txt',
                               transform=transforms.Compose([Normalizer(), KittiResizer()]),
                               categories = CATEGORY)
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet = torch.load(parser.model_path)


    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    print(kitti_eval.evaluate(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold)))



if __name__ == '__main__':
    main()
