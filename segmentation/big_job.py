import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import XinguDataset
from dataset.onehotencoding import OneHotEncoding
from datetime import datetime
import glob
import time
import numpy as np

BATCH_SIZE = 16
NUM_EPOCHS = 2
NUM_FOLDS = 5
PATCH_SIZE = 256
STRIDE_SIZE = 64
STYLE = 'Discard'
compositions = {
    "False Color Urban": [7, 6, 4],
    "RGB": [4, 3, 2],
    "Allbands": range(1, 8),
    "Color Infrared": [5, 4, 3],
    "Vegetative Analysis": [6, 5, 4],
    "Shortwave Infrared": [7, 5, 4],
    "GA-156": [1, 5, 6],
    "GA-1456": [1, 4, 5, 6],
    "GA-146": [1, 4, 6]
}

regions_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
folds = np.array_split(regions_list, NUM_FOLDS)

for COMPOSITION in compositions:
    for fold in range(0, NUM_FOLDS):
        train_regions = np.concatenate(
            [regions for j, regions in enumerate(folds) if j != fold])
        test_regions = folds[fold]
        # regions = np.random.choice(range(1, 11), size=10, replace=False)
        CHANNELS = len(compositions[COMPOSITION])
        # (model, loss, lr)
        configs = [(smp.DeepLabV3Plus(in_channels=CHANNELS,
                                      classes=1,
                                      activation='softmax'),
                    smp.utils.losses.JaccardLoss(), 1e-3),
                   (smp.FPN(in_channels=CHANNELS,
                            classes=2,
                            activation='softmax'),
                    smp.utils.losses.CrossEntropyLoss(), 1e-3),
                   (smp.Linknet(in_channels=CHANNELS,
                                classes=1,
                                activation='softmax'),
                    smp.utils.losses.DiceLoss(), 1e-4),
                   (smp.MAnet(in_channels=CHANNELS,
                              classes=2,
                              activation='softmax'),
                    smp.utils.losses.DiceLoss(), 1e-4),
                   (smp.PAN(in_channels=CHANNELS,
                            classes=1,
                            activation='softmax'),
                    smp.utils.losses.JaccardLoss(), 1e-3),
                   (smp.PSPNet(in_channels=CHANNELS,
                               classes=1,
                               activation='softmax'),
                    smp.utils.losses.DiceLoss(), 1e-3),
                   (smp.Unet(in_channels=CHANNELS,
                             classes=1,
                             activation='softmax'),
                    smp.utils.losses.JaccardLoss(), 1e-4),
                   (smp.UnetPlusPlus(in_channels=CHANNELS,
                                     classes=1,
                                     activation='softmax'),
                    smp.utils.losses.DiceLoss(), 1e-3)]
        for (model, loss, lr) in configs:
            best_epoch = 0
            max_f1 = 0
            max_precision = 0
            max_iou = 0
            max_accuracy = 0
            max_recall = 0

            encoder = OneHotEncoding(2)

            print(f"{10 * '#'} {model.__class__.__name__} {10*'#'}")
            # instantiating datasets
            train_ds = XinguDataset('./dataset/scenes_allbands',
                                    './dataset/truth_masks',
                                    encoder,
                                    compositions[COMPOSITION],
                                    train_regions,
                                    PATCH_SIZE,
                                    STRIDE_SIZE,
                                    transforms=True)
            test_ds = XinguDataset('./dataset/scenes_allbands',
                                   './dataset/truth_masks', encoder,
                                   compositions[COMPOSITION], test_regions,
                                   PATCH_SIZE, STRIDE_SIZE)

            writer = SummaryWriter(
                log_dir=
                f"runs/{STYLE}-{model.__class__.__name__}_{loss.__class__.__name__}_Adam_{lr :.0e}-{COMPOSITION}-fold-{fold}"
            )

            optimizer = torch.optim.Adam([
                dict(params=model.parameters(), lr=lr),
            ])

            metrics = [
                smp.utils.metrics.IoU(),
                smp.utils.metrics.Fscore(),
                smp.utils.metrics.Precision(),
                smp.utils.metrics.Accuracy(),
                smp.utils.metrics.Recall()
            ]

            train_epoch = smp.utils.train.TrainEpoch(
                model,
                loss=loss,
                metrics=metrics,
                optimizer=optimizer,
                device='cuda',
                verbose=True,
            )
            test_epoch = smp.utils.train.ValidEpoch(
                model,
                loss=loss,
                metrics=metrics,
                device='cuda',
                verbose=True,
            )

            # dataloaders for this fold
            train_dataloader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=BATCH_SIZE,
                drop_last=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_ds,
                batch_size=4,
                drop_last=True,
            )

            start = time.time()
            torch.cuda.reset_max_memory_allocated()

            for epoch in range(0, NUM_EPOCHS):
                print(f'\nEpoch: {epoch}')
                train_logs = train_epoch.run(train_dataloader)
                test_logs = test_epoch.run(test_dataloader)

                if max_iou < test_logs['iou_score']:
                    torch.save(
                        model,
                        f'./models/{STYLE}-{model.__class__.__name__}-{COMPOSITION}.pth'
                    )
                    max_iou = test_logs['iou_score']
                    max_precision = test_logs['precision']
                    max_f1 = test_logs['fscore']
                    max_accuracy = test_logs['accuracy']
                    max_recall = test_logs['recall']
                    best_epoch = epoch
                    print('Model saved!')

                # gathering data
                loss_train = next(iter(train_logs.values()))
                iou_score_train = train_logs['iou_score']

                precision_test = test_logs['precision']
                f1_test = test_logs['fscore']
                iou_score_test = test_logs['iou_score']
                accuracy_test = test_logs['accuracy']
                recall_test = test_logs['recall']

                # writing to tensorboard
                writer.add_scalar(f'Loss/train', loss_train, epoch)
                writer.add_scalar(f'IoU/train', iou_score_train, epoch)
                writer.add_scalar(f'Precision/test', precision_test, epoch)
                writer.add_scalar(f'F1/test', f1_test, epoch)
                writer.add_scalar(f'Jaccard/test', iou_score_test, epoch)
                writer.add_scalar(f'Accuracy/test', accuracy_test, epoch)
                writer.add_scalar(f'Recall/test', recall_test, epoch)
                writer.add_scalar(f'Max Metrics/F1', max_f1, epoch)
                writer.add_scalar(f'Max Metrics/IoU', max_iou, epoch)
                writer.add_scalar(f'Max Metrics/Precision', max_precision,
                                  epoch)
                writer.add_scalar(f'Max Metrics/Accuracy', max_accuracy, epoch)

            end = time.time()
            execution_time = end - start

            # Convert execution time to minutes and seconds
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)

            max_memory_usage = torch.cuda.max_memory_reserved() / (1024**2)
            with open(
                    f'results/results-{STYLE}-{model.__class__.__name__}-{lr :.0e}-{COMPOSITION}-fold-{fold}.txt',
                    'w') as f:
                f.write("TEST RESULTS\n")
                f.write(
                    f'{model.__class__.__name__}-{COMPOSITION}-fold-{fold}\n')
                f.write(f'Precision: {max_precision :.4f}\n')
                f.write(f'F1 Score: {max_f1 :.4f}\n')
                f.write(f'IoU: {max_iou :.4f}\n')
                f.write(f'Accuracy: {max_accuracy :.4f}\n')
                f.write(f'Recall: {max_recall :.4f}\n')
                f.write(f'On epoch: {best_epoch}\n')
                f.write(f'Time: {minutes}m {seconds}s\n')
                f.write(f'Memória (MiB): {max_memory_usage}\n')

            writer.close()
