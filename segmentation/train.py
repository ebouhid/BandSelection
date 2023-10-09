import torch
from dataset.dataset import XinguDataset
import models
import pytorch_lightning as pl
import sys
from mlflowmodelcheckpoint import MLFlowModelCheckpoint

# Set experiment name
INFO = 'Local_FromScratch'

# Get model name as command line argument
MODEL_NAME = str(sys.argv[1])

# Set hyperparameters
BATCH_SIZE = 64
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
    model = models.SegmentationModelsPytorch_PL(model_name=MODEL_NAME,
                                                in_channels=len(
                                                    compositions[COMPOSITION]),
                                                num_classes=NUM_CLASSES)

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

    # Instantiating logger
    logger = pl.loggers.MLFlowLogger(experiment_name=INFO)

    # Log model hyperparameters
    logger.log_hyperparams({"model": model.__class__.__name__})
    logger.log_hyperparams({"composition": COMPOSITION})
    logger.log_hyperparams({"batch_size": BATCH_SIZE})
    logger.log_hyperparams({"num_epochs": NUM_EPOCHS})
    logger.log_hyperparams({"patch_size": PATCH_SIZE})
    logger.log_hyperparams({"stride_size": STRIDE_SIZE})
    logger.log_hyperparams({"num_classes": NUM_CLASSES})
    logger.log_hyperparams({"train_regions": train_regions})
    logger.log_hyperparams({"test_regions": test_regions})

    # Instantiating trainer
    checkpoint_callback = MLFlowModelCheckpoint(mlflow_logger=logger,
                                                monitor='val_iou',
                                                verbose=True)
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                         logger=logger,
                         callbacks=[checkpoint_callback])

    # Training
    trainer.fit(model, train_loader, test_loader)
