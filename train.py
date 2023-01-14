import argparse
import collections

import numpy as np
from shutil import rmtree
import os 

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import KittiDataset, CocoDataset, CSVDataset, collater, KittiResizer, AspectRatioBasedSampler, HorizontalFlipping, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from retinanet import kitti_eval

assert torch.__version__.split('.')[0] == '1'

# TODO, for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

BATCH_SIZE = 8
NUMBER_WORKERS = 8
EXP_NAME = "2D_detection"
CATEGORY = ['Car']

OUTPUT_DIR = f"checkpoint/{EXP_NAME}"
print("Clean output directory : " + OUTPUT_DIR)
rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

# TODO load pre-trained model weight
PATH_TO_WEIGHTS = "checkpoint/2D_detection/epoch3.pt"

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='kitti')
    parser.add_argument('--kitti_path', help='Path to KITTI directory', default='/home/lab530/KenYu/kitti/')
    parser.add_argument('--split_path', help='Path to KITTI directory', default='/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/')
    
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=30) #  default=100

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'kitti':
        dataset_train = KittiDataset(parser.kitti_path, split_path=f'{parser.split_path}train.txt',
                                     transform=transforms.Compose([Normalizer(), HorizontalFlipping(), KittiResizer()]),
                                     categories = CATEGORY)
        dataset_val   = KittiDataset(parser.kitti_path, split_path=f'{parser.split_path}val.txt',
                                     transform=transforms.Compose([Normalizer(), KittiResizer()]),
                                     categories = CATEGORY)
    elif parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), HorizontalFlipping(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), HorizontalFlipping(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=BATCH_SIZE, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=NUMBER_WORKERS, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=NUMBER_WORKERS, collate_fn=collater, batch_sampler=sampler_val)
        # TODO can dataloader_val be faster?
    
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Load model weight
    if os.path.exists(PATH_TO_WEIGHTS):
        print(f"Use pretrained model at {PATH_TO_WEIGHTS}")
        retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    else:
        print(f"Cannot find pretrain model at {PATH_TO_WEIGHTS}")

    retinanet = retinanet.cuda()
    retinanet = torch.nn.DataParallel(retinanet).cuda()
    
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # print(data['img'].shape) # torch.Size([8, 3, 384, 1280])
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | cls_loss: {:1.5f} | reg_loss: {:1.5f} | total_loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        
        if parser.dataset == 'kitti':
            print('Evaluating dataset')
            mAP = kitti_eval.evaluate(dataset_val, retinanet)

        elif parser.dataset == 'coco':
            print('Evaluating dataset')
            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:
            print('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, f'{OUTPUT_DIR}/epoch{epoch_num}.pt')

    retinanet.eval()

    torch.save(retinanet, f'{OUTPUT_DIR}/model_final.pt')


if __name__ == '__main__':
    main()
