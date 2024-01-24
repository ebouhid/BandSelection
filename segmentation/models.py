import segmentation_models_pytorch
import pytorch_lightning as pl
from segmentation_models_pytorch.utils.losses import BCELoss
import torch
import torchmetrics
import numpy as np
import hashlib


class SegmentationModelsPytorch_PL(pl.LightningModule):
    def __init__(self,
                 model_name,
                 in_channels,
                 num_classes,
                 activation='sigmoid',
                 encoder_name='resnet101',
                 encoder_weights=None):
        super().__init__()

        # Defining model
        self.model_name = model_name
        model_class = getattr(segmentation_models_pytorch, model_name)
        self.model = model_class(in_channels=in_channels,
                                 classes=num_classes,
                                 activation=activation,
                                 encoder_name=encoder_name,
                                 encoder_weights=encoder_weights)

        self.loss = BCELoss()
        self.lr = 1e-3

        # Defining metrics
        self.train_accuracy = torchmetrics.Accuracy(task='binary')
        self.val_accuracy = torchmetrics.Accuracy(task='binary')
        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')
        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')
        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(),
                                     lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=10,
                                                    gamma=0.9)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and training metrics
        loss = self.loss(outputs, targets)
        train_accuracy = np.float64(self.train_accuracy(outputs, targets))
        train_precision = np.float64(self.train_precision(outputs, targets))
        train_recall = np.float64(self.train_recall(outputs, targets))
        train_f1 = np.float64(self.train_f1(outputs, targets))
        train_iou = np.float64(self.train_iou(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', train_accuracy, on_epoch=True)
        self.log('train_precision', train_precision, on_epoch=True)
        self.log('train_recall', train_recall, on_epoch=True)
        self.log('train_f1', train_f1, on_epoch=True)
        self.log('train_iou', train_iou, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss and validation metrics
        loss = self.loss(outputs, targets)
        val_accuracy = np.float64(self.val_accuracy(outputs, targets))
        val_precision = np.float64(self.val_precision(outputs, targets))
        val_recall = np.float64(self.val_recall(outputs, targets))
        val_f1 = np.float64(self.val_f1(outputs, targets))
        val_iou = np.float64(self.val_iou(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True)
        self.log('val_precision', val_precision, on_epoch=True)
        self.log('val_recall', val_recall, on_epoch=True)
        self.log('val_f1', val_f1, on_epoch=True)
        self.log('val_iou', val_iou, on_epoch=True)
