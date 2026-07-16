import torch
from torchvision.ops import nms as torchvision_nms


def apply_nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long)

    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    scores = torch.as_tensor(scores, dtype=torch.float32)

    return torchvision_nms(boxes, scores, iou_threshold)