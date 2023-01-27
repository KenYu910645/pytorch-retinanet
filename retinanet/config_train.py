DEVICE = 'cuda:0' # 'cuda:1', 'cuda:0' 

BATCH_SIZE = 4
NUMBER_WORKERS = 8
NUM_EPOCH = 100
VALID_EPOCH = 1 # 5
SAVE_EPOCH  = 5 # 5
EXP_NAME = "2D_detection_fine_tune"
# TODO all_feature_map currently has no use
BACKBONE = "original" # "original" 'crop_feature_map' 'all_feature_map' 'crop_feature_map_fuse_16'

CATEGORY = ['Car']
OUTPUT_DIR = f"checkpoint/{EXP_NAME}"
PATH_TO_WEIGHTS = "/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_original_pretrain.pt"
SPLIT_PATH = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/debug_split/" # "only_one_split " "dummy_exp" "chen_split" "debug_split"
