from __future__ import print_function

import numpy as np
import torch

import sys
sys.path.append("/home/lab530/KenYu/ml_toolkit/kitti/kitti_eval")
from main_eval_from_gac import eval_from_gac

def _get_detections(dataset, retinanet, save_path, device, score_threshold=0.05, max_detections=100):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    # all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale_h, scale_w = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).to(device).float().unsqueeze(dim=0))
           
            # select indices which have a score above the threshold
            indices = torch.nonzero(scores > score_threshold, as_tuple=False).squeeze(dim=1)

            #
            # detections = []
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]
                # find the order with which to sort the scores
                _, scores_sort = torch.sort(scores, descending=True)[:max_detections]
                
                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                
                # Untransform detection boudning box
                image_boxes[:, 0] /= scale_w
                image_boxes[:, 1] /= scale_h
                image_boxes[:, 2] /= scale_w
                image_boxes[:, 3] /= scale_h
                
                #                 
                image_detections = torch.cat((image_boxes,
                                              image_scores.view(-1, 1),
                                              image_labels.view(-1, 1)), axis=1)
                # copy detections to all_detections
                # for label in range(dataset.num_classes()):
                #     all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1].cpu().numpy()
                # detections.append( image_detections[image_detections[:, -1] == 0, :-1].cpu().numpy() )
                detections = image_detections[image_detections[:, -1] == 0, :-1].cpu().numpy()
            else:
                detections =  np.zeros((0, 5))
                #     # copy detections to all_detections
                #     # for label in range(dataset.num_classes()):
                #     #     all_detections[index][label] = np.zeros((0, 5))
                #     detections.append( np.zeros((0, 5)) )

            # Output prection result to .txt
            with open(f"{save_path}/{dataset.img_names[index]}.txt", 'w') as f:
                # Only output car detection
                s = ""
                if len(detections) > 0:
                    for x1, y1, x2, y2, score in detections:
                        # 'category, truncated, occluded alpha, xmin, ymin, xmax, ymax, height, width, length, x3d, y3d, z3d, rot_y, score]
                        # 1         2          3        4      5     6     7     8     9       10     11      12   13   14   15     16
                        s += f"Car -1000 -1000 -1000 {x1} {y1} {x2} {y2} -1000 -1000 -1000 -1000 -1000 -1000 -1000 {score}\n"
                f.write(s)
            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

def evaluate(
    generator,
    retinanet,
    save_path,
    split_path,
    device,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save precision recall curve of each label.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections 
    _get_detections(generator, retinanet, save_path, device, score_threshold=score_threshold, max_detections=max_detections)
    
    try:
        result_txt = eval_from_gac(label_path="/home/lab530/KenYu/kitti/training/label_2",
                                result_path=save_path,
                                label_split_file=split_path + "val.txt",
                                current_classes=[0],
                                gpu=device,
                                dataset_type='kitti')
    except Exception as e:
        print(e)
        return "Can't evaluate on validation set"
    return result_txt[0]

