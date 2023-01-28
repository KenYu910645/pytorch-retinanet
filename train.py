import argparse
import collections

import numpy as np
from shutil import rmtree
import os 

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import KittiDataset, CocoDataset, CSVDataset, kitti_collater, KittiResizer, AspectRatioBasedSampler, HorizontalFlipping, \
    Normalizer
from torch.utils.data import DataLoader, BatchSampler, RandomSampler
from retinanet import coco_eval
from retinanet import csv_eval
from retinanet import kitti_eval

from retinanet.config_train import *

assert torch.__version__.split('.')[0] == '1'

# for debugging
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

print("Clean output directory : " + OUTPUT_DIR)
rmtree(OUTPUT_DIR, ignore_errors=True)
os.mkdir(OUTPUT_DIR)

# 
SAVE_PATH = os.path.join(OUTPUT_DIR, 'result')
print("Clean output directory : " + SAVE_PATH)
rmtree(SAVE_PATH, ignore_errors=True)
os.mkdir(SAVE_PATH)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='kitti')
    parser.add_argument('--kitti_path', help='Path to KITTI directory', default='/home/lab530/KenYu/kitti/')
    # parser.add_argument('--split_path', help='Path to KITTI directory', default='/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/debug_split/')
    
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    # parser.add_argument('--epochs', help='Number of epochs', type=int, default=30) #  default=100

    parser = parser.parse_args(args)


    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, mode = BACKBONE, device = DEVICE)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
    
    # Load model weight
    if os.path.exists(PATH_TO_WEIGHTS):
        print(f"Use pretrained model at {PATH_TO_WEIGHTS}")

        checkpoint = torch.load(PATH_TO_WEIGHTS)
        
        # Load the whole weight 
        # retinanet.load_state_dict(checkpoint['model_state_dict'])
        
        # Load partial of the pre-train model
        try:
            pretrained_dict = checkpoint['model_state_dict']
        except KeyError: 
            pretrained_dict = checkpoint
        
        model_dict = retinanet.state_dict()
        # Reference: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        retinanet.load_state_dict(model_dict)
        
    else:
        print(f"Cannot find pretrain model at {PATH_TO_WEIGHTS}")
    
    retinanet = retinanet.to(DEVICE)
    # retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5) # 1e-5
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True) # 3
    # TODO, try cosineAnneling
    
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.freeze_bn() # retinanet.module.freeze_bn()

    # Create the data loaders
    if parser.dataset == 'kitti':
        # TODO, i temporary disable hisrozontal flipping for simplisity
        dataset_train = KittiDataset(parser.kitti_path, split_path=f'{SPLIT_PATH}train.txt',
                                     transform=transforms.Compose([Normalizer(), KittiResizer()]),
                                     categories = CATEGORY)
        dataset_val   = KittiDataset(parser.kitti_path, split_path=f'{SPLIT_PATH}val.txt',
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

    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=BATCH_SIZE, drop_last=False)
    # sampler_train = BatchSampler(RandomSampler(dataset_train), batch_size=8, drop_last=True)
    # dataloader_train = DataLoader(dataset_train, num_workers=NUMBER_WORKERS, collate_fn=collater, batch_sampler=sampler_train)
    # dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUMBER_WORKERS, shuffle=True)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=NUMBER_WORKERS, 
                                  shuffle=True, pin_memory=True, collate_fn = kitti_collater)


    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(NUM_EPOCH):

        retinanet.train()
        retinanet.freeze_bn() # retinanet.module.freeze_bn()

        epoch_loss = []

        # print(next(retinanet.parameters()).device) # cuda:1
        for iter_num, data in enumerate(dataloader_train):
            # print(data['img'].shape) # torch.Size([8, 3, 384, 1280])
            optimizer.zero_grad()
            
            classification_loss, regression_loss = retinanet([data['img'].to(DEVICE).float(), 
                                                              data['annot'].to(DEVICE)])

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

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            
            print(
                'Epoch: {} | Iteration: {} | Learning Rate: {:1.5f}*e^-5 | cls_loss: {:1.5f} | reg_loss: {:1.5f} | total_loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(lr*100000), float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        if (epoch_num + 1) % SAVE_EPOCH == 0:
            # torch.save(retinanet.module, f'{OUTPUT_DIR}/epoch{epoch_num}.pt')
            # torch.save(retinanet, f'{OUTPUT_DIR}/epoch{epoch_num}.pt')
            torch.save({
                        'model_state_dict': retinanet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f'{OUTPUT_DIR}/epoch{epoch_num}.pt')

            print(f"Saved checkpoint to {OUTPUT_DIR}/epoch{epoch_num}.pt")

        if (epoch_num + 1) % VALID_EPOCH == 0:
            if parser.dataset == 'kitti':
                print('Evaluating dataset')
                # mAP = kitti_eval.evaluate(dataset_val, retinanet, SAVE_PATH)
                mAP = kitti_eval.evaluate(dataset_val, 
                                            retinanet, 
                                            SAVE_PATH,
                                            SPLIT_PATH,
                                            DEVICE, 
                                            iou_threshold = IOU_THRESHOLD)
                print(mAP)
                with open( f'{OUTPUT_DIR}/epoch{epoch_num}_val_result.txt', "w") as f:
                    f.write(mAP)
            
            elif parser.dataset == 'coco':
                print('Evaluating dataset')
                coco_eval.evaluate_coco(dataset_val, retinanet)

            elif parser.dataset == 'csv' and parser.csv_val is not None:
                print('Evaluating dataset')
                mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

    retinanet.eval()

    # torch.save(retinanet, f'{OUTPUT_DIR}/model_final.pt')


if __name__ == '__main__':
    main()
