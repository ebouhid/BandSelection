import torch
from dataset.dataset import XinguDataset
import models
import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow

# Set experiment name
INFO = 'LightningTest'
mlflow.set_experiment(INFO)

# Get model name as command line argument
MODEL_NAME = str(sys.argv[1])

# Set hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
NUM_CLASSES = 1
DATASET_DIR = './dataset/scenes_allbands_ndvi'
GT_DIR = './dataset/truth_masks'

# Set compositions
compositions = {
    "4": [4],
    "43": [4, 3],
    "431": [4, 3, 1],
    "4316": [4, 3, 1, 6],
    "4317": [4, 3, 1, 7],
    "43167": [4, 3, 1, 6, 7],
    "All+NDVI": range(1, 9),
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
                            GT_DIR,
                            compositions[COMPOSITION],
                            train_regions,
                            patch_size=PATCH_SIZE,
                            stride_size=STRIDE_SIZE,
                            reflect_pad=False,
                            transforms=False)
    test_ds = XinguDataset(DATASET_DIR,
                           GT_DIR,
                           compositions[COMPOSITION],
                           test_regions,
                           patch_size=PATCH_SIZE,
                           stride_size=PATCH_SIZE,
                           reflect_pad=True,
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
    mlflow.pytorch.autolog()
    mlflow.log_params({
        'model_name': MODEL_NAME,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patch_size': PATCH_SIZE,
        'stride_size': STRIDE_SIZE,
        'num_classes': NUM_CLASSES,
        'dataset_dir': DATASET_DIR,
        'gt_dir': GT_DIR,
        'composition': COMPOSITION,
        'train_regions': train_regions,
        'test_regions': test_regions,
        'train_size': len(train_ds),
        'test_size': len(test_ds)
    })

    # Instantiating checkpoint callback
    checkpoint_callback = ModelCheckpoint(dirpath='./models/', filename=f'{INFO}-{MODEL_NAME}-{COMPOSITION}', monitor='val_iou', save_top_k=1, mode='max', verbose=True, save_last=True)

    # Instantiating trainer
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                         callbacks=[checkpoint_callback])

    # Training
    trainer.fit(model, train_loader, test_loader)
