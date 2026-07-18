## Coin Detection

Pipeline for detecting, localizing and classifying nine denominations of coins using specialized models for background filtering, bounding box refinement and denomination recognition.


## Custom detector

Detection consists of three stages:

1. `BackgroundClassifier` rejects image crops without coins.
2. `DifferenceRegressor` adjusts the bounding box coordinates.
3. `MoneyClassifier` predicts the coin denomination.

## Results

The models were compared on 18 validation images containing 177 coins.

| Model | mAP@0.5 | mAP@0.5:0.95 | Time per image |
|---|---:|---:|---:|
| CustomDetector | 0.781 | 0.575 | 925 ms |
| YOLO26n | 0.851 | 0.776 | 31 ms |

After 50 epochs, YOLO achieved higher accuracy and was approximately 30 times faster than the custom detector.

## Project structure

```text
datasets/    dataset loading and preprocessing
models/      model definitions and the complete detector
notebooks/   training, evaluation and model comparison
utils/       non-maximum suppression
data/        training and validation data
```

## Usage

Run the notebooks in the following order:

1. `train_background.ipynb`
2. `train_regressor.ipynb`
3. `train_classifier.ipynb`
4. `evaluate_detector.ipynb`
5. `compare_yolo_custom.ipynb`
