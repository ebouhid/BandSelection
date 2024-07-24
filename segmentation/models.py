import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from copy import deepcopy
import cv2
import os
from general_balanced import GBC, functional_gbc


class DeforestationDetectionModel(pl.LightningModule):
    def __init__(self, in_channels, composition_name, loss, encoder_name='resnet101', lr=1e-3, encoder_weights='imagenet', debug=False, **kwargs):
        super().__init__()

        # Defining model
        self.model = smp.DeepLabV3Plus(in_channels=in_channels,
                                       classes=1,
                                       activation='sigmoid',
                                       encoder_name=encoder_name,
                                       encoder_weights=encoder_weights)

        self.loss = loss
        self.lr = lr

        self.alpha = None
        self.beta = None
        self.gamma = None

        self.composition_name = composition_name
        self.debug = debug

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
            print(f'Setting {kwarg} to {kwargs[kwarg]}!')

        if loss.__class__.__name__ == 'FocalLoss':
            self.alpha = self.loss.alpha
            self.gamma = self.loss.gamma
        elif loss.__class__.__name__ == 'BinaryTverskyLoss':
            self.alpha = self.loss.alpha
            self.beta = self.loss.beta

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
        self.train_gbc = GBC(k=10)
        self.val_gbc = GBC(k=10)

    def forward(self, x):
        return self.model(x)

    def set_fold_info(self, fold_num, train_regions, test_regions):
        self.fold_num = fold_num
        self.train_regions = [f"x{region :02d}" for region in train_regions]
        self.test_regions = [f"x{region :02d}" for region in test_regions]

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=5, factor=0.9, mode='min', verbose=True),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
            'name': 'lr_scheduler'
        }

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
        train_gbc = np.float64(self.train_gbc(outputs, targets))

        # Log metrics
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.log('train_accuracy', train_accuracy,
                 on_epoch=True, sync_dist=True)
        self.log('train_precision', train_precision,
                 on_epoch=True, sync_dist=True)
        self.log('train_recall', train_recall, on_epoch=True, sync_dist=True)
        self.log('train_f1', train_f1, on_epoch=True, sync_dist=True)
        self.log('train_iou', train_iou, on_epoch=True, sync_dist=True)
        self.log('train_gbc', train_gbc, on_epoch=True, sync_dist=True)

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
        val_gbc = np.float64(self.val_gbc(outputs, targets))

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_accuracy', val_accuracy, on_epoch=True, sync_dist=True)
        self.log('val_precision', val_precision, on_epoch=True, sync_dist=True)
        self.log('val_recall', val_recall, on_epoch=True, sync_dist=True)
        self.log('val_f1', val_f1, on_epoch=True, sync_dist=True)
        self.log('val_iou', val_iou, on_epoch=True, sync_dist=True)
        self.log('val_gbc', val_gbc, on_epoch=True, sync_dist=True)

        return val_f1

    def on_save_checkpoint(self, checkpoint):
        if not hasattr(self, "scenes_dir"):
            print(f"No scenes_dir attribute set to {self.__class__.__name__}, will not save predictions.")
            return
        elif not hasattr(self, "truth_dir"):
            print(f"No truth_dir attribute set to {self.__class__}, will not save predictions.")
            return
        elif (not hasattr(self, "train_regions")) or (not hasattr(self, "test_regions")):
            print(f"Missing fold information in {self.__class__}, will not save predictions. Be sure to use the set_fold_info() method.")
            return

        stride = 256
        patch_size = (256, 256)

        if self.fold_num is None:
            pred_dir = 'predictions'
            test_regions = self.test_regions
        else:
            pred_dir = f'predictions/fold{self.fold_num}'
            test_regions = self.test_regions

        os.makedirs(pred_dir, exist_ok=True)
        # Get images and patchify
        for region in test_regions:
            image = np.load(
                os.path.join(self.scenes_dir, f'{region}.npy'))
            # Normalize image
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

            truth = np.load(os.path.join(self.truth_dir, f"truth_{region}.npy")).squeeze()
            # Adjust to binary segmentation
            truth = np.where(truth == 2, 0, 1)

            height, width, _ = image.shape
            newwidth = math.ceil(width / stride) * stride
            newheight = math.ceil(height / stride) * stride

            composition = self.composition_name
            loss = self.loss.__class__.__name__

            bands = [
                int(i) for i in composition] if composition != "All+NDVI" else list(range(1, 10))
            bands = [i - 1 for i in bands]
            image = image[:, :, bands]

            # Patchify image
            patchified_image = patchify(image, patch_size, stride)
            image_patches = patchified_image["patches"]
            image_patchcounts = patchified_image["patch_counts"]

            # Load model
            model = deepcopy(self).eval()

            # Iterate through patches and perform predictions
            predicted_masks = []
            for patch, (x, y) in zip(image_patches, image_patchcounts):
                patch = torch.tensor(patch, device='cuda').permute(
                    2, 0, 1).unsqueeze(0).float()
                with torch.no_grad():
                    prediction = model(patch)
                    # prediction = torch.sigmoid(prediction) # Sigmoid is already applied in the model
                    if self.debug:
                        print(
                            f'Prediction range: {prediction.min()} - {prediction.max()}')
                    prediction = (prediction > 0.5)
                predicted_masks.append((prediction, (x, y)))
                # cv2.imwrite(f'{pred_dir}/{region}_{composition}_{loss}_{x}_{y}.png', prediction.squeeze().cpu().numpy() * 255)

            # Stitch patches together
            stitched_mask = np.zeros((newheight, newwidth), dtype=np.uint8)
            for mask, (x, y) in predicted_masks:
                mask = mask.squeeze().cpu().numpy()
                stitched_mask[x:x + patch_size[0], y:y + patch_size[1]] = mask
            
            # Clip to original size
            stitched_mask = stitched_mask[:height, :width]

            # Build confusion mask
            confusion_mask = np.zeros((height, width, 3))
            true_positive_color = (1, 1, 1)
            false_positive_color = (0, 0, 1)
            false_negative_color = (1, 0, 0)
            true_negative_color = (0, 0, 0)

            # Save stitched mask
            np.save(f'{pred_dir}/{region}_{composition}_{loss}.npy', stitched_mask)

            assert stitched_mask.shape == truth.shape, f"Shapes don't match: {stitched_mask.shape} and {truth.shape}"

            true_positives = np.logical_and(
                stitched_mask == 1, truth == 1)
            false_positives = np.logical_and(
                stitched_mask == 1, truth == 0)
            false_negatives = np.logical_and(
                stitched_mask == 0, truth == 1)
            true_negatives = np.logical_and(
                stitched_mask == 0, truth == 0)
            confusion_mask[true_positives] = true_positive_color
            confusion_mask[false_positives] = false_positive_color
            confusion_mask[false_negatives] = false_negative_color
            confusion_mask[true_negatives] = true_negative_color

            # Calculate metrics for whole image
            true_positives = np.sum(true_positives)
            false_positives = np.sum(false_positives)
            false_negatives = np.sum(false_negatives)
            true_negatives = np.sum(true_negatives)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * (precision * recall) / (precision + recall)
            accuracy = (true_positives + true_negatives) / (true_positives +
                                                            false_positives + false_negatives + true_negatives)
            iou = true_positives / \
                (true_positives + false_positives + false_negatives)
            gbc = functional_gbc(true_positives, true_negatives,
                                 false_positives, false_negatives)

            # Extend lower part of confusion mask for writing text
            if self.fold_num is not None:
                vertical_pad = 300
                fold_str = f"Fold: {self.fold_num}"
                train_regions_str = f"Train Regions: {self.train_regions}"
                test_regions_str = f"Test Regions: {self.test_regions}"
            else:
                vertical_pad = 200
            confusion_mask = np.pad(confusion_mask, ((
                0, vertical_pad), (0, 75), (0, 0)), mode='constant', constant_values=0)
            confusion_mask = (confusion_mask * 255).astype(np.uint8)

            # Extend right part of image for writing text
            if confusion_mask.shape[1] < 1280:
                confusion_mask = np.pad(confusion_mask, ((0, 0), (0, 1280 - confusion_mask.shape[1]), (0, 0)), mode='constant', constant_values=0)

            # Write metrics on image
            metrics_str = f"Precision: {precision :.2f} | Recall: {recall :.2f} | F1: {f1 :.2f} | Accuracy: {accuracy :.2f} | IoU: {iou :.2f}"
            gbc_str = f"GBC: {gbc :.2f}"
            model_info_str = f"Composition: {composition} | Loss: {loss}"
            filename = f'{pred_dir}/{region}_{composition}_{loss}'
            if self.alpha is not None:
                model_info_str += f" | Alpha: {self.alpha}"
                filename += f"_alpha{str(self.alpha).replace('.', '')}"
            if self.beta is not None:
                model_info_str += f" | Beta: {self.beta}"
                filename += f"_beta{str(self.beta).replace('.', '')}"
            if self.gamma is not None:
                model_info_str += f" | Gamma: {self.gamma}"
                filename += f"_gamma{str(self.gamma).replace('.', '')}"
            if self.fold_num is not None:
                filename += f"_fold{self.fold_num}"
            filename += '.png'

            # Calculate font size based on confusion_mask size
            font_size = confusion_mask.shape[0] / 1000 * 0.8
            thickness = int(font_size * 2)
            cv2.putText(confusion_mask, metrics_str, (0, height + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            cv2.putText(confusion_mask, gbc_str, (0, height + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            cv2.putText(confusion_mask, model_info_str, (0, height + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            cv2.putText(confusion_mask, fold_str, (0, height + 155),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            cv2.putText(confusion_mask, train_regions_str, (0, height + 200),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            cv2.putText(confusion_mask, test_regions_str, (0, height + 245),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), thickness)
            confusion_mask = cv2.cvtColor(confusion_mask, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, confusion_mask)

            # Also save information to a .txt file
            with open(f'{pred_dir}/{region}_{composition}_{loss}.txt', 'w') as f:
                f.write(metrics_str + '\n')
                f.write(gbc_str + '\n')
                f.write(model_info_str + '\n')
                f.write(fold_str + '\n')
                f.write(train_regions_str + '\n')
                f.write(test_regions_str + '\n')


def patchify(array, patch_size, stride):
    height, width, _ = array.shape
    patches = []
    patch_counts = []
    wholesize = (math.ceil(height / stride) * stride,
                 math.ceil(width / stride) * stride)
    stitched_array = np.zeros(
        (wholesize[0], wholesize[1], array.shape[2]), dtype=array.dtype)
    for x in range(0, height, stride):
        for y in range(0, width, stride):
            # Crop the patch from the input image
            patch = array[x:x + patch_size[0], y:y + patch_size[1], :]
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                # print(f'Padding patch at {x}, {y} with shape {patch.shape}')
                bottompad = patch_size[0] - patch.shape[0]
                rightpad = patch_size[1] - patch.shape[1]
                patch = np.pad(
                    patch, ((0, bottompad), (0, rightpad), (0, 0)), mode='reflect')
            patches.append(patch)
            patch_counts.append((x, y))
            stitched_array[x:x + patch_size[0], y:y + patch_size[1], :] = patch

    return {"patches": patches, "patch_counts": patch_counts, "stitched": stitched_array}
