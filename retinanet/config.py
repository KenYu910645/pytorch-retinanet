DEVICE = 'cuda:0' # 'cuda:1', 'cuda:0' 

BATCH_SIZE = 8
NUMBER_WORKERS = 8
EXP_NAME = "2D_detection"
CATEGORY = ['Car']

OUTPUT_DIR = f"checkpoint/{EXP_NAME}"
PATH_TO_WEIGHTS = "/home/lab530/KenYu/pytorch-retinanet/checkpoint/epoch4.pt"
SPLIT_PATH = "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/debug_split/" # "/home/lab530/KenYu/visualDet3D/visualDet3D/data/kitti/debug_split/"

VALID_EPOCH = 5 # 5
SAVE_EPOCH  = 1 # 5