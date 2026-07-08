import torch


BACKGROUND_CLASS = 9


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_label_of_crop(crop_box, boxes, labels, iou_threshold):
    best_iou = 0.0
    best_idx = None

    for i, gt_box in enumerate(boxes):
        iou = calculate_iou(crop_box, gt_box.tolist())

        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx is not None and best_iou >= iou_threshold:
        label = int(labels[best_idx])
        bb_diff = boxes[best_idx] - torch.tensor(crop_box, dtype=torch.float32)

        return label, bb_diff

    return BACKGROUND_CLASS, torch.zeros(4)