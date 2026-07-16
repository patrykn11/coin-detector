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