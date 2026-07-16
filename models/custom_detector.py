import torch

from models.background_classifier import BackgroundClassifier
from models.difference_regressor import DifferenceRegressor
from models.money_classifier import MoneyClassifier


class CustomDetector:
    def __init__(self, background_checkpoint, regressor_checkpoint, classifier_checkpoint, device="mps", crop_size=64, stride=32):
        self.device = torch.device(device)

        self.crop_size = crop_size
        self.stride = stride

        self.background_model = BackgroundClassifier().to(self.device)
        self.regressor_model = DifferenceRegressor().to(self.device)
        self.classifier_model = MoneyClassifier().to(self.device)

        self.background_model.load_state_dict(torch.load(background_checkpoint, map_location=self.device, weights_only=True))
        self.regressor_model.load_state_dict(torch.load(regressor_checkpoint, map_location=self.device, weights_only=True))
        self.classifier_model.load_state_dict(torch.load(classifier_checkpoint, map_location=self.device, weights_only=True))

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

        return image[:, y1:y1 + self.crop_size, x1:x1 + self.crop_size]