import torch
from data.dataset import XinguDataset
from data.cross_val_iterator import CrossValidationIterator
import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
import albumentations as A
import general_balanced as gb
import argparse

# Set experiment name
INFO = 'GBCLoss_Comparison_SGD'
mlflow.set_experiment(INFO)

# Set hyperparameters
MODEL_NAME = 'DeepLabV3Plus'
BATCH_SIZE = 16
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
NUM_CLASSES = 1
DATASET_DIR = './data/scenes_allbands_ndvi'
GT_DIR = './data/truth_masks'
COMPOSITION = [6, 5, 1]
compname = '' + ''.join([str(i) for i in COMPOSITION]
                        ) if COMPOSITION != range(1, 10) else "All+NDVI"

# Set regions
regions = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # Regions from 1 to 10 (excluding 5)

loss = gb.GBCLoss(k=10)
# loss = torch.nn.BCEWithLogitsLoss()

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 *
                           0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])

# Instantiate KFold
kfold = CrossValidationIterator(regions)

for fold, (train_regions, test_regions) in enumerate(kfold):
    compname = f'{compname}'

    train_ds = XinguDataset(DATASET_DIR,
                            GT_DIR,
                            COMPOSITION,
                            train_regions,
                            patch_size=PATCH_SIZE,
                            stride_size=STRIDE_SIZE,
                            reflect_pad=True,
                            transforms=aug)
    test_ds = XinguDataset(DATASET_DIR,
                           GT_DIR,
                           COMPOSITION,
                           test_regions,
                           patch_size=PATCH_SIZE,
                           stride_size=PATCH_SIZE,
                           reflect_pad=True,
                           transforms=False)

    # Instantiating dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=16,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=16)

    # Instantiate the model for each fold
    model = models.DeforestationDetectionModel(in_channels=len(
        COMPOSITION), composition_name=compname, loss=loss)
    model.set_fold_info(fold, train_regions, test_regions)

    # Instantiating logger
    mlflow.pytorch.autolog()
    mlflow.log_params({
        'model_name': MODEL_NAME,
        'loss': model.loss,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patch_size': PATCH_SIZE,
        'stride_size': STRIDE_SIZE,
        'num_classes': NUM_CLASSES,
        'dataset_dir': DATASET_DIR,
        'gt_dir': GT_DIR,
        'composition': compname,
        'train_regions': train_regions,
        'test_regions': test_regions,
        'train_size': len(train_ds),
        'test_size': len(test_ds)
    })

    # Instantiating checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath='./models/', filename=f'{INFO}-{MODEL_NAME}-{compname}-fold{fold}', monitor='val_iou', save_top_k=1, mode='max', verbose=True)

    # Instantiating trainer
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                         callbacks=[checkpoint_callback], accelerator="gpu", devices=[0, 1])

    # Training
    trainer.fit(model, train_loader, test_loader)
    mlflow.end_run()
