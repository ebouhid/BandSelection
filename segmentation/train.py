import torch
from segmentation_dataset.dataset import XinguDataset
from segmentation_dataset.cross_val_iterator import CrossValidationIterator
import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import mlflow
import albumentations as A
import general_balanced as gb
import argparse

def set_args():
    parser = argparse.ArgumentParser(description="Training script for DeepLabV3Plus model.")
    parser.add_argument('--model_name', type=str, default='DeepLabV3Plus', help='Name of the model to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for training.')
    parser.add_argument('--stride_size', type=int, default=64, help='Stride size for patch extraction.')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes.')
    parser.add_argument('--dataset_dir', type=str, default='./data/scenes_allbands_ndvi', help='Path to the dataset directory.')
    parser.add_argument('--gt_dir', type=str, default='./data/truth_masks', help='Path to the ground truth masks directory.')
    parser.add_argument('--composition', type=int, nargs='+', default=[6, 5, 1], help='Composition of input data.')
    parser.add_argument("--loss", type=str,choices=['bce', 'bccelogits', 'gbcloss'], default='bce')
    parser.add_argument("--info", type=str, required=False, help='Additional information to be added to the model name.')
    parser.add_argument("--exp_name", type=str, default='Default')
    parser.add_argument("--gpu_ids", type=int, nargs='+', required=False, help='GPU IDs to use.')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate for training.')

    return parser.parse_args()

args = set_args()

# Assign parsed arguments to original variable names
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
PATCH_SIZE = args.patch_size
STRIDE_SIZE = args.stride_size
NUM_CLASSES = args.num_classes
DATASET_DIR = args.dataset_dir
GT_DIR = args.gt_dir
COMPOSITION = args.composition
LOSS_FN = args.loss
INFO = args.info
GPU_IDS = args.gpu_ids if args.gpu_ids is not None else "auto"
LR = args.lr

# Set experiment name
mlflow.set_experiment(args.exp_name)

compname = '' + ''.join([str(i) for i in COMPOSITION]
                        ) if COMPOSITION != range(1, 10) else "All+NDVI"

# Set regions
regions = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # Regions from 1 to 10 (excluding 5)

if LOSS_FN == 'bce':
    loss = torch.nn.BCELoss()
elif LOSS_FN == 'bcelogits':
    loss = torch.nn.BCEWithLogitsLoss()
elif LOSS_FN == 'gbcloss':
    loss = gb.GBCLoss(k=10)

aug = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
])

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
    model = models.DeforestationDetectionModel(in_channels=len(COMPOSITION),
                                               composition_name=compname,
                                               loss=loss,
                                               lr=LR,
                                               info=INFO,
                                               scenes_dir=DATASET_DIR,
                                               truth_dir=GT_DIR)
    
    model.set_fold_info(fold, train_regions, test_regions)

    # Instantiating logger
    mlflow.pytorch.autolog()
    mlflow.log_params({
        'model_name': MODEL_NAME,
        'gpu_ids': GPU_IDS,
        'loss': model.loss,
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'patch_size': PATCH_SIZE,
        'stride_size': STRIDE_SIZE,
        'num_classes': NUM_CLASSES,
        'dataset_dir': DATASET_DIR,
        'gt_dir': GT_DIR,
        'info': INFO,
        'lr': LR,
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
                         callbacks=[checkpoint_callback], accelerator="gpu", devices=GPU_IDS)

    # Training
    trainer.fit(model, train_loader, test_loader)
    mlflow.end_run()
