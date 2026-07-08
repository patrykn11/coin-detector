import random
import torch

from datasets.coin_crop_dataset import CoinCropDataset


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

        bb_diff = boxes[best_idx] - torch.tensor(
            crop_box,
            dtype=torch.float32
        )

        return label, bb_diff

    return BACKGROUND_CLASS, torch.zeros(4)


def prepare_dataset(
    dataset,
    config,
    num_of_crops_per_image=64,
    box_size=64,
):
    offset = config["offset"]
    bg_percent = config["bg_percent"]
    iou_threshold = config["iou_threshold"]

    crops = []
    crop_labels = []
    bb_diffs = []

    for sample in dataset:
        image = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]

        _, height, width = image.shape

        for i in range(num_of_crops_per_image):
            sample_background = (
                random.random() < bg_percent
                or len(boxes) == 0
            )

            if sample_background:
                x1 = random.randint(0, width - box_size)
                y1 = random.randint(0, height - box_size)

            else:
                idx = i % len(boxes)
                gt_box = boxes[idx]

                cx = (gt_box[0] + gt_box[2]) / 2
                cy = (gt_box[1] + gt_box[3]) / 2

                cx += random.randint(-offset, offset)
                cy += random.randint(-offset, offset)

                x1 = int(cx - box_size / 2)
                y1 = int(cy - box_size / 2)

                x1 = max(0, min(x1, width - box_size))
                y1 = max(0, min(y1, height - box_size))

            x2 = x1 + box_size
            y2 = y1 + box_size

            crop = image[:, y1:y2, x1:x2]
            crop_box = [x1, y1, x2, y2]

            label, bb_diff = get_label_of_crop(
                crop_box,
                boxes,
                labels,
                iou_threshold
            )

            crops.append(crop)
            crop_labels.append(label)
            bb_diffs.append(bb_diff)

    return CoinCropDataset(crops, crop_labels, bb_diffs)