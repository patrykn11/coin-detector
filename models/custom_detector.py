import torch

from torchvision.transforms.functional import to_tensor

from models.background_classifier import BackgroundClassifier
from models.difference_regressor import DifferenceRegressor
from models.money_classifier import MoneyClassifier

from utils.nms import apply_nms


class CustomDetector:
    def __init__(
        self,
        background_checkpoint,
        regressor_checkpoint,
        classifier_checkpoint,
        device="mps",
        crop_size=64,
        stride=32,
    ):
        self.device = torch.device(device)

        self.crop_size = crop_size
        self.stride = stride

        self.background_model = BackgroundClassifier().to(self.device)
        self.regressor_model = DifferenceRegressor().to(self.device)
        self.classifier_model = MoneyClassifier().to(self.device)

        self.background_model.load_state_dict(
            torch.load(
                background_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.regressor_model.load_state_dict(
            torch.load(
                regressor_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.classifier_model.load_state_dict(
            torch.load(
                classifier_checkpoint,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.background_model.eval()
        self.regressor_model.eval()
        self.classifier_model.eval()

    def centered_crop(self, image, box):
        _, height, width = image.shape

        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        x1 = int(cx - self.crop_size / 2)
        y1 = int(cy - self.crop_size / 2)

        x1 = max(0, min(x1, width - self.crop_size))
        y1 = max(0, min(y1, height - self.crop_size))

        return image[
            :,
            y1:y1 + self.crop_size,
            x1:x1 + self.crop_size,
        ]

    @torch.inference_mode()
    def predict(self, image):
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image.convert("RGB"))

        image = image.to(self.device)

        _, height, width = image.shape

        detections = []

        for y1 in range(0, height - self.crop_size + 1, self.stride):
            for x1 in range(0, width - self.crop_size + 1, self.stride):
                x2 = x1 + self.crop_size
                y2 = y1 + self.crop_size

                crop = image[:, y1:y2, x1:x2].unsqueeze(0)

                # BackgroundClassifier
                bg_logit = self.background_model(crop)

                probability = torch.sigmoid(bg_logit).item()

                if probability <= 0.5:
                    continue

                # DifferenceRegressor
                diff = self.regressor_model(crop).squeeze(0)

                crop_box = torch.tensor(
                    [x1, y1, x2, y2],
                    dtype=torch.float32,
                    device=self.device,
                )

                corrected_box = crop_box + diff

                corrected_box[0::2] = corrected_box[0::2].clamp(0, width)
                corrected_box[1::2] = corrected_box[1::2].clamp(0, height)

                if (
                    corrected_box[2] <= corrected_box[0]
                    or corrected_box[3] <= corrected_box[1]
                ):
                    continue

                # MoneyClassifier
                corrected_crop = self.centered_crop(
                    image,
                    corrected_box,
                ).unsqueeze(0)

                logits = self.classifier_model(corrected_crop)

                probabilities = torch.softmax(logits, dim=1).squeeze(0)

                label = int(probabilities.argmax().item())

                if label == 9:
                    continue

                confidence = float(probabilities[label].item())

                detections.append(
                    (
                        corrected_box.cpu().tolist(),
                        label,
                        confidence,
                    )
                )

        if not detections:
            return []

        boxes = [d[0] for d in detections]
        labels = [d[1] for d in detections]
        scores = [d[2] for d in detections]

        keep = apply_nms(boxes, scores, iou_threshold=0.3)

        results = []

        for i in keep.tolist():
            x1, y1, x2, y2 = boxes[i]

            results.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    labels[i],
                    scores[i],
                )
            )

        return results

    def __call__(self, image):
        return self.predict(image)