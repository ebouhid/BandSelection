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
            precision_list = []
            recall_list = []
            f1_list = []
            accuracy_list = []
            iou_list = []
            for region in regions:

                if composition == "PCA":
                    image = Image.open(f'dataset/scenes_pca/pca_{region}.png')
                    image = np.array(image, dtype=np.float32)
                else:
                    image = np.load(f'dataset/scenes_allbands_ndvi/allbands_ndvi_{region}.npy')[:, :, compositions[composition]]
                
                truth_mask = np.load(f'dataset/truth_masks/truth_{region}.npy')
                
                height, width, _ = image.shape

                model = torch.load(f'models/UMDA_Compositions-DeepLabV3Plus-{composition}.pth')

                model.eval()

                predicted_masks = []
                patch_counts = []

                # Iterate through patches
                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        # Crop the patch from the input image
                        patch = image[y:y + patch_size[1], x:x + patch_size[0], :]
                        if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                            bottompad = patch_size[0] - patch.shape[0]
                            rightpad = patch_size[1] - patch.shape[1]
                            patch = np.pad(patch, ((0, bottompad), (0, rightpad), (0, 0)), mode='reflect')
                        
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
                        truth_mask_patch = truth_mask[y:y+patch_size[1], x:x+patch_size[0]].squeeze()
                        if truth_mask_patch.shape != predicted_mask.shape:
                            # print(f'Patch shape: {predicted_mask.shape}')
                            # print(f'Truth mask shape: {truth_mask_patch.shape}')
                            bottompad = patch_size[0] - truth_mask_patch.shape[0]
                            rightpad = patch_size[1] - truth_mask_patch.shape[1]
                            truth_mask_patch = np.pad(truth_mask_patch, ((0, bottompad), (0, rightpad)), mode='reflect')
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
        
            accuracy_avg = np.mean(accuracy_list)
            precision_avg = np.mean(precision_list)
            recall_avg = np.mean(recall_list)
            f1_avg = np.mean(f1_list)
            iou_avg = np.mean(iou_list)

            # print(f"Composition: {composition}")
            # print(f'Accuracy list: {accuracy_list}')
            # print(f'Precision list: {precision_list}')
            # print(f'Recall list: {recall_list}')
            # print(f'F1 list: {f1_list}')
            # print(f'IoU list: {iou_list}')
            print(f'total patches: {len(accuracy_list)}')

            # Write in latex format
            f.write(f"{composition} & {precision_avg * 100:.2f} & {recall_avg * 100:.2f} & {f1_avg * 100:.2f} & {accuracy_avg * 100:.2f} & {iou_avg * 100:.2f} \\ \hline \n")

            # Also add to a pandas dataframe
            df.append({
                "Composition": composition,
                "Precision": precision_avg * 100,
                "Recall": recall_avg * 100,
                "F1": f1_avg * 100,
                "Accuracy": accuracy_avg * 100,
                "IoU": iou_avg * 100
            })

        df = pd.DataFrame(df)
        df = df.sort_values(by="IoU", ascending=False)  
        df.to_excel("outputs/results_byhand.xlsx", index=False)
