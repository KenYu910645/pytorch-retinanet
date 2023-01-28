DEVICE = 'cuda:0' # 'cuda:1', 'cuda:0' 

BATCH_SIZE = 8
NUMBER_WORKERS = 8
NUM_EPOCH = 3000
VALID_EPOCH = 100 # 5
SAVE_EPOCH  = 999999 # 5
EXP_NAME = "2D_detection_fine_tune"
# TODO all_feature_map currently has no use
BACKBONE = "original" # "original" 'crop_feature_map' 'all_feature_map' 'crop_feature_map_fuse_16'
IOU_THRESHOLD = 0.5

CATEGORY = ['Car']
OUTPUT_DIR = f"checkpoint/{EXP_NAME}"
PATH_TO_WEIGHTS = "asdf" # "/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_original_pretrain.pt" # "/home/lab530/KenYu/pytorch-retinanet/checkpoint/coco_resnet_50_map_0_335_state_dict.pt" # "/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_original_pretrain.pt"
SPLIT_PATH = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/only_one_split/" # "only_one_split " "dummy_exp" "chen_split" "debug_split"
