import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    regions = ["x03", "x04"]

    patch_size = (256, 256)
    stride = 256

    df = []
    
    compositions = {
        "All+NDVI": range(1, 9),
        "4": [4],
        "43": [4, 3],
        "431": [4, 3, 1],
        "4317": [4, 3, 1, 7],
        "4316": [4, 3, 1, 6],
        "43167": [4, 3, 1, 6, 7],
        # "PCA": range(1, 4)
    }

    # Convert to 0-based indexing
    for composition in compositions:
        compositions[composition] = [x - 1 for x in compositions[composition]]

    # Open a .txt file to store the results in a LaTeX table
    with open('outputs/latex_table.txt', 'w') as f:
        f.write("\\textbf{Composition} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} & \\textbf{Accuracy} & \\textbf{IoU} & \\textbf{Epoch}\\ \hline\n")
        for composition in compositions:
            for region in regions:
                precision_list = []
                recall_list = []
                f1_list = []
                accuracy_list = []
                iou_list = []

                if composition == "PCA":
                    image = Image.open(f'dataset/scenes_pca/pca_{region}.png')
                    image = np.array(image, dtype=np.float32)
                else:
                    image = np.load(f'dataset/scenes_allbands_ndvi/allbands_ndvi_{region}.npy')[:, :, compositions[composition]]
                
                truth_mask = np.load(f'dataset/truth_masks/truth_{region}.npy')
                
                height, width, _ = image.shape

                model = torch.load(f'models/ShortScale-DeepLabV3Plus-{composition}.pth')

                model.eval()

                predicted_masks = []
                patch_counts = []

                # Iterate through patches
                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        # Crop the patch from the input image
                        patch = image[y:y + patch_size[1], x:x + patch_size[0], :]
                        # Check for dimensions
                        if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                            rightpad = patch_size[0] - patch.shape[1]
                            bottompad = patch_size[1] - patch.shape[0]
                            patch = np.pad(patch, ((0, bottompad),(0, rightpad), (0, 0)), mode='reflect')
                        patch = torch.tensor((np.transpose(patch, (2, 0, 1))/ 255.0), device='cuda')

                        # Make predictions using the model
                        with torch.no_grad():
                            predicted_mask = model(patch.unsqueeze(0))  # Add batch dimension
                            # print(predicted_mask.shape)

                        # Convert the predicted mask to a numpy array
                        predicted_mask = predicted_mask.squeeze().cpu().numpy()

                        # Store the predicted mask and patch count
                        predicted_masks.append(predicted_mask)
                        patch_counts.append((x, y))

                        predicted_mask = np.where(predicted_mask > 0.5, 1, 0)

                        # Identify true positives, true negatives, false positives, and false negatives for the patch
                        truth_mask_patch = truth_mask[y:y+patch_size[1], x:x+patch_size[0]]
                        if truth_mask_patch.shape[0] != patch_size[0] or truth_mask_patch.shape[1] != patch_size[1]:
                            rightpad = patch_size[0] - truth_mask_patch.shape[1]
                            bottompad = patch_size[1] - truth_mask_patch.shape[0]
                            truth_mask_patch = np.pad(truth_mask_patch, ((0, bottompad),(0, rightpad), (0, 0)), mode='reflect')
                        
                        true_positives = np.logical_and(truth_mask_patch == 1, predicted_mask == 1)
                        true_negatives = np.logical_and(truth_mask_patch == 0, predicted_mask == 0)
                        false_positives = np.logical_and(truth_mask_patch == 0, predicted_mask == 1)
                        false_negatives = np.logical_and(truth_mask_patch == 1, predicted_mask == 0)

                        # Calculate the number of true positives, true negatives, false positives, and false negatives
                        tp = np.sum(true_positives)
                        tn = np.sum(true_negatives)
                        fp = np.sum(false_positives)
                        fn = np.sum(false_negatives)

                        # Calculate metrics
                        true_positives = np.sum(true_positives)
                        true_negatives = np.sum(true_negatives)
                        false_positives = np.sum(false_positives)
                        false_negatives = np.sum(false_negatives)

                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        precision = tp / (tp + fp)
                        recall = tp / (tp + fn)
                        f1_score = 2 * precision * recall / (precision + recall)
                        iou = tp / (tp + fp + fn)
                        
                        accuracy_list.append(accuracy)
                        precision_list.append(precision)
                        recall_list.append(recall)
                        f1_list.append(f1_score)
                        iou_list.append(iou)

                # Create an empty result image with the same size as the input image
                concatenated_pred = np.zeros((y + patch_size[1], x + patch_size[0]))

                # Create an empty mask with the same size as the patches
                mask_accumulator = np.zeros((y + patch_size[1], x + patch_size[0]))

                # Combine the predicted masks into the result image
                for mask, (x, y) in zip(predicted_masks, patch_counts):
                    x1, x2 = x, x + patch_size[0]
                    y1, y2 = y, y + patch_size[1]

                    # Clip the patch and mask to fit within the image boundaries
                    x1, x2 = max(0, x1), min(width, x2)
                    y1, y2 = max(0, y1), min(height, y2)

                    # Calculate the clipped patch dimensions
                    patch_width = x2 - x1
                    patch_height = y2 - y1

                    # Calculate the corresponding mask region
                    mask_region = mask[:patch_height, :patch_width]

                    # Add the mask region to the accumulator
                    # print(f'x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}')
                    # print(f'mask_accumulator.shape: {mask_accumulator.shape}')
                    mask_accumulator[y1:y2, x1:x2] += mask_region

                # Might add a confusion mask generation block here
        
            accuracy_avg = np.mean(accuracy_list)
            precision_avg = np.mean(precision_list)
            recall_avg = np.mean(recall_list)
            f1_avg = np.mean(f1_list)
            iou_avg = np.mean(iou_list)

            print(f"Composition: {composition}")
            print(f'Accuracy list: {accuracy_list}')
            print(f'Precision list: {precision_list}')
            print(f'Recall list: {recall_list}')
            print(f'F1 list: {f1_list}')
            print(f'IoU list: {iou_list}')

            # Write in latex format
            f.write(f"{composition} & {precision_avg * 100:.2f} & {recall_avg * 100:.2f} & {f1_avg * 100:.2f} & {accuracy_avg * 100:.2f} & {iou_avg * 100:.2f} \\ \hline \n")

            # Also add to a pandas dataframe
            df.append({
                "Composition": [composition],
                "Precision": [precision_avg * 100],
                "Recall": [recall_avg * 100],
                "F1": [f1_avg * 100],
                "Accuracy": [accuracy_avg * 100],
                "IoU": [iou_avg * 100]
            })

        df = pd.DataFrame(df)
        df.to_csv("outputs/results.csv")
