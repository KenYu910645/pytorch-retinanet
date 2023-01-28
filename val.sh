EXP_NAME=('/home/lab530/KenYu/pytorch-retinanet/checkpoint/2D_detection_original_pretrain.pt')

for exp_name in "${EXP_NAME[@]}"
do
    python kitti_validation.py --weights "$exp_name" --device cuda:0 --backbone original
done