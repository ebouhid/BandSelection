import mlflow
import torch
from dataset.dataset import XinguDataset
from datetime import datetime
import glob
import time
import os
import models
import pytorch_lightning as pl
# Set experiment name
INFO = 'Local_FromScratch'
os.environ['MLFLOW_EXPERIMENT_NAME'] = INFO

# Set hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
NUM_CLASSES = 1
DATASET_DIR = './dataset/scenes_allbands_ndvi'
TRUTH_DIR = './dataset/truth_masks'

# Set compositions
compositions = {
    "Allbands": range(1, 8),
    "RGB": [4, 3, 2],
    "6": [6],
    "65": [6, 5],
    "651": [6, 5, 1],
    "6514": [6, 5, 1, 4],
    "6517": [6, 5, 1, 7],
}

# Set regions
train_regions = [1, 2, 6, 7, 8, 9, 10]  # Do not use region 5 anywhere
test_regions = [3, 4]

for COMPOSITION in compositions:
    model = models.DeepLabV3Plus_PL(in_channels=len(compositions[COMPOSITION]),
                                    num_classes=NUM_CLASSES)
    with mlflow.start_run():
        # Log model hyperparameters
        mlflow.log_param("model", model.__class__.__name__)
        mlflow.log_param("composition", COMPOSITION)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("patch_size", PATCH_SIZE)
        mlflow.log_param("stride_size", STRIDE_SIZE)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("train_regions", train_regions)
        mlflow.log_param("test_regions", test_regions)

        # Instantiating datasets
        train_ds = XinguDataset(DATASET_DIR,
                                TRUTH_DIR,
                                compositions[COMPOSITION],
                                train_regions,
                                patch_size=PATCH_SIZE,
                                stride_size=STRIDE_SIZE,
                                transforms=True)
        test_ds = XinguDataset(DATASET_DIR,
                               TRUTH_DIR,
                               compositions[COMPOSITION],
                               test_regions,
                               patch_size=PATCH_SIZE,
                               stride_size=PATCH_SIZE,
                               transforms=False)

        # Instantiating dataloaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=False,
                                                  num_workers=8)

        # Instantiating trainer
        trainer = pl.Trainer(max_epochs=NUM_EPOCHS)

        # Training
        trainer.fit(model, train_loader, test_loader)
