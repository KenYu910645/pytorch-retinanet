from retinanet.anchors import Anchors
import torch
 
anchor = Anchors()

# print(f"img_batch = {img_batch.shape}") # torch.Size([1, 3, 384, 1280])
img_batch = torch.zeros((1, 3, 384, 1280))

a = anchor(img_batch)
