import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smpu
import mlflow
from dataset.dataset_pca import XinguDataset
from datetime import datetime
import glob
import time
import numpy as np
import os
import threading
from gpuprofiling import track_gpu

BATCH_SIZE = 64
NUM_EPOCHS = 100
PATCH_SIZE = 256
STRIDE_SIZE = 64
INFO = 'Test_tracking'
NUM_CLASSES = 1
TRACKING_INTERVAL = 0.1

os.environ['MLFLOW_EXPERIMENT_NAME'] = INFO

compositions = {"PCA": range(1, 4)}

train_regions = [1, 2, 6, 7, 8, 9, 10]  # Do not use region 5 anywhere
test_regions = [3, 4]
for COMPOSITION in compositions:
    TRACKING_FNAME = f'./results/{INFO}-{COMPOSITION}-power.csv'
    # Start GPU power draw tracking
    stop_event = threading.Event()
    tracking_thread = threading.Thread(target=track_gpu,
                                       args=(TRACKING_INTERVAL, TRACKING_FNAME,
                                             stop_event))
    tracking_thread.start()

    CHANNELS = len(compositions[COMPOSITION])
    # (model, loss, lr)
    configs = [
        (smp.Unet(
            in_channels=CHANNELS,
            classes=NUM_CLASSES,
            activation='sigmoid',
            encoder_name='resnet34',
            encoder_weights=None,
        ), smp.utils.losses.JaccardLoss(), 5e-4),
    ]

    for (model, loss, lr) in configs:
        with mlflow.start_run():
            best_epoch = 0
            max_f1 = 0
            max_precision = 0
            max_iou = 0
            max_accuracy = 0
            max_recall = 0

            print(f"{10 * '#'} {model.__class__.__name__} {10*'#'}")
            # instantiating datasets
            train_ds = XinguDataset('./dataset/scenes_pca',
                                    './dataset/truth_masks',
                                    compositions[COMPOSITION],
                                    train_regions,
                                    PATCH_SIZE,
                                    STRIDE_SIZE,
                                    transforms=True)
            test_ds = XinguDataset('./dataset/scenes_pca',
                                   './dataset/truth_masks',
                                   compositions[COMPOSITION],
                                   test_regions,
                                   PATCH_SIZE,
                                   PATCH_SIZE,
                                   transforms=False)

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
                drop_last=False,
                shuffle=True,
                num_workers=8)

            test_dataloader = torch.utils.data.DataLoader(
                dataset=test_ds,
                batch_size=BATCH_SIZE,
                drop_last=False,
                shuffle=False,
                num_workers=8)

            # logging parameters
            mlflow.log_params({
                "model": model.__class__.__name__,
                "loss": loss.__class__.__name__,
                "lr": lr,
                "composition": COMPOSITION,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "patch_size": PATCH_SIZE,
                "stride_size": STRIDE_SIZE,
                "Description": INFO,
                "train_regions": train_regions,
                "test_regions": test_regions
            })

            start = time.time()
            torch.cuda.reset_max_memory_allocated()

            for epoch in range(1, NUM_EPOCHS + 1):
                print(f'\nEpoch: {epoch}')
                train_logs = train_epoch.run(train_dataloader)
                test_logs = test_epoch.run(test_dataloader)

                if max_iou < test_logs['iou_score']:
                    torch.save(
                        model,
                        f'./models/{INFO}-{model.__class__.__name__}-{COMPOSITION}.pth'
                    )
                    torch.save(
                        model.state_dict(),
                        f'./models/{INFO}-{model.__class__.__name__}-{COMPOSITION}-StateDict.pth'
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

                # logging to mlflow
                mlflow.log_metric('train_loss', loss_train, epoch)
                mlflow.log_metric('train_iou', iou_score_train, epoch)
                mlflow.log_metric('test_precision', precision_test, epoch)
                mlflow.log_metric('test_f1', f1_test, epoch)
                mlflow.log_metric('test_iou', iou_score_test, epoch)
                mlflow.log_metric('test_accuracy', accuracy_test, epoch)
                mlflow.log_metric('test_recall', recall_test, epoch)

            end = time.time()
            execution_time = end - start
            stop_event.set()
            tracking_thread.join()

            # Convert execution time to minutes and seconds
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)

            max_memory_usage = torch.cuda.max_memory_reserved() / (1024**2)
            with open(
                    f'results/results-{INFO}-{model.__class__.__name__}-{lr :.0e}-{COMPOSITION}.txt',
                    'w') as f:
                f.write("TEST RESULTS\n")
                f.write(f'{model.__class__.__name__}-{COMPOSITION}\n')
                f.write(f'Precision: {max_precision :.4f}\n')
                f.write(f'F1 Score: {max_f1 :.4f}\n')
                f.write(f'IoU: {max_iou :.4f}\n')
                f.write(f'Accuracy: {max_accuracy :.4f}\n')
                f.write(f'Recall: {max_recall :.4f}\n')
                f.write(f'On epoch: {best_epoch}\n')
                f.write(f'Time: {minutes}m {seconds}s\n')
