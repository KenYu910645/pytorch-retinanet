DEVICE = 'cuda:1' # 'cuda:1', 'cuda:0' 

BATCH_SIZE = 8
NUMBER_WORKERS = 8
NUM_EPOCH = 100
VALID_EPOCH = 9999 # 5
SAVE_EPOCH  = 5 # 5
EXP_NAME = "2D_detection_crop_feature_map_fuse_16"
# TODO all_feature_map currently has no use
BACKBONE = "crop_feature_map_fuse_16" # "original" 'crop_feature_map' 'all_feature_map'

CATEGORY = ['Car']
OUTPUT_DIR = f"checkpoint/{EXP_NAME}"
PATH_TO_WEIGHTS = "/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_2/epoch29.pt"
SPLIT_PATH = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/chen_split/" # "only_one_split " "dummy_exp" "chen_split" "debug_split"
