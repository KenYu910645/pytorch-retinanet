EXP_NAME=('checkpoint/2D_detection_crop_feature_map/epoch19.pt' 'checkpoint/2D_detection_crop_feature_map/epoch59.pt' 'checkpoint/2D_detection_crop_feature_map/epoch99.pt' 'checkpoint/2D_detection_2/epoch159.pt' 'checkpoint/2D_detection_2/epoch239.pt' 'checkpoint/2D_detection_2/epoch359.pt')

for exp_name in "${EXP_NAME[@]}"
do
    python kitti_validation.py --weights "$exp_name" --device cuda:0
done