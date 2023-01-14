import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import KittiResizer, Normalizer, KittiDataset
from retinanet import kitti_eval
from retinanet.config import CATEGORY, DEVICE, OUTPUT_DIR, SPLIT_PATH
import os
from shutil import rmtree
assert torch.__version__.split('.')[0] == '1'

# 
SAVE_PATH = os.path.join(OUTPUT_DIR, 'result')
print("Clean output directory : " + SAVE_PATH)
rmtree(SAVE_PATH, ignore_errors=True)
os.mkdir(SAVE_PATH)

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='kitti')
    parser.add_argument('--kitti_path', help='Path to KITTI directory', default='/home/lab530/KenYu/kitti/')
    # parser.add_argument('--split_path', help='Path to KITTI directory', default='/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/debug_split/')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)

    parser.add_argument('--model_path', help='Path to model', type=str, default="/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection/epoch6.pt")
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    
    
    parser = parser.parse_args(args)

    # Load dataset
    dataset_val = KittiDataset(parser.kitti_path, split_path=f'{SPLIT_PATH}val.txt',
                               transform=transforms.Compose([Normalizer(), KittiResizer()]),
                               categories = CATEGORY)
    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    # retinanet = retinanet.to(DEVICE)
    # retinanet = torch.load(parser.model_path)

    # Load model weight
    if os.path.exists(parser.model_path):
        print(f"Load weight at {parser.model_path}")
        checkpoint = torch.load(parser.model_path)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Cannot find weight at {parser.model_path}")

    retinanet = retinanet.to(DEVICE)
    # retinanet = retinanet.cuda()
    # retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn() # retinanet.module.freeze_bn()

    print(kitti_eval.evaluate(dataset_val, retinanet, iou_threshold = float(parser.iou_threshold), save_path = SAVE_PATH))



if __name__ == '__main__':
    main()
