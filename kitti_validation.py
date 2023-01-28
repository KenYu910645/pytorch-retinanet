import argparse
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import KittiResizer, Normalizer, KittiDataset
from retinanet import kitti_eval
import os
from shutil import rmtree
assert torch.__version__.split('.')[0] == '1'

# BACKBONE = "crop_feature_map" # "original" 'crop_feature_map' 'all_feature_map'
SPLIT_PATH = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/" # "only_one_split " "dummy_exp" "chen_split" "debug_split"
KITTI_PATH = "/home/lab530/KenYu/kitti/"
CATEGORY = ['Car']
IOU_THRESHOLD = 0.5

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--weights',type=str, default='/path/to/weights.pt')
    parser.add_argument('--backbone',type=str, default='original')
    parser.add_argument('--device' ,type=str, default='cuda:0')
    parser = parser.parse_args(args)
    
    # Clean result.txt directory
    # SAVE_PATH = os.path.join(OUTPUT_DIR, 'result')
    save_path = os.path.join(os.path.split(parser.weights)[0], os.path.split(parser.weights)[1].split('.')[0] + "_result") 
    print("Clean output directory : " + save_path)
    rmtree(save_path, ignore_errors=True)
    os.mkdir(save_path)
    

    # Create the model
    retinanet = model.resnet50(num_classes=len(CATEGORY), pretrained=True, mode = parser.backbone, device = parser.device)

    # Load model weight
    if os.path.exists(parser.weights):
        print(f"Load weight at {parser.weights}")
        checkpoint = torch.load(parser.weights, map_location=parser.device)
        retinanet.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Cannot find weight at {parser.weights}")
        raise ValueError

    retinanet = retinanet.to(parser.device)
    # retinanet = retinanet.cuda()
    # retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn() # retinanet.module.freeze_bn()

    # Load dataset
    dataset_val = KittiDataset(KITTI_PATH, split_path=f'{SPLIT_PATH}val.txt',
                               transform=transforms.Compose([Normalizer(), KittiResizer()]),
                               categories = CATEGORY)

    result_str = kitti_eval.evaluate(dataset_val, 
                                    retinanet, 
                                    save_path,
                                    SPLIT_PATH,
                                    parser.device, 
                                    iou_threshold = IOU_THRESHOLD)
    print(result_str)
    with open( parser.weights.split(".")[0] + "_val_result.txt", "w") as f:
        f.write(result_str)

if __name__ == '__main__':
    main()
