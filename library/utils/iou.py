import torch

# smooth is for avoiding 0/0
SMOOTH = 1e-6

def iou_pytorch(preds, labels):

    intersection = (preds & labels).float().sum((1, 2))  
    union = (preds | labels).float().sum((1, 2))         
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  
    return thresholded