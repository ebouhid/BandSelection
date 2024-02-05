import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

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
    }

    # Convert to 0-based indexing
    for composition in compositions:
        compositions[composition] = [x - 1 for x in compositions[composition]]

    # Create a directory to store the confusion masks
    os.makedirs('outputs/confusion_masks', exist_ok=True)

    # Open a .txt file to store the results in a LaTeX table
    with open('outputs/latex_table.txt', 'w') as f:
        f.write("\\textbf{Composition} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} & \\textbf{Accuracy} & \\textbf{IoU} & \\textbf{Epoch}\\ \hline\n")
        for region in regions:
            df = []
            for composition in compositions:
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

                model = torch.load(f'models/UMDA_Compositions_Imagenet-DeepLabV3Plus-{composition}.pth')

                model.eval()

                predicted_masks = []
                patch_counts = []
                if region == "x03":
                    paddedsize = (1024, 1024)
                if region == "x04":
                    paddedsize = (1024, 768)

                confusion_mask_scene = np.zeros((paddedsize[0], paddedsize[1], 3), dtype=np.uint8)
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
                        
                        # Create confusion mask for patch
                        confusion_mask = np.zeros((patch_size[1], patch_size[0], 3), dtype=np.uint8)
                        confusion_mask[true_positives] = [255, 255, 255]
                        confusion_mask[true_negatives] = [0, 0, 0]
                        confusion_mask[false_positives] = [0, 0, 255]
                        confusion_mask[false_negatives] = [255, 0, 0]

                        # Confusion mask for whole scene
                        confusion_mask_scene[y:y+patch_size[1], x:x+patch_size[0], :] = confusion_mask

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

                        # Write metrics for this patch on the image
                        plt.imshow(confusion_mask)
                        plt.text(10, 10, f'Accuracy: {accuracy:.2f}', color='lime')
                        plt.text(10, 30, f'Precision: {precision:.2f}', color='lime')
                        plt.text(10, 50, f'Recall: {recall:.2f}', color='lime')
                        plt.text(10, 70, f'F1: {f1_score:.2f}', color='lime')
                        plt.text(10, 90, f'IoU: {iou:.2f}', color='lime')
                        plt.savefig(f'outputs/confusion_masks/confusion_{composition}_{region}_{x}_{y}.png')
                        plt.clf()
        
                accuracy_avg = np.mean(accuracy_list)
                precision_avg = np.mean(precision_list)
                recall_avg = np.mean(recall_list)
                f1_avg = np.mean(f1_list)
                iou_avg = np.mean(iou_list)

                print(f'total patches: {len(iou_list)}')
                # # Save the confusion mask for the whole scene
                # os.makedirs('outputs/big_confusion_masks', exist_ok=True)
                # confusion_mask_scene = Image.fromarray(confusion_mask_scene)
                # confusion_mask_scene.save(f'outputs/big_confusion_masks/bigconfusion_{composition}_{region}.png')
                assert iou_avg == np.sum(iou_list) / len(iou_list)

            # # Write in latex format
            # f.write(f"{composition} & {precision_avg * 100:.2f} & {recall_avg * 100:.2f} & {f1_avg * 100:.2f} & {accuracy_avg * 100:.2f} & {iou_avg * 100:.2f} \\ \hline \n")

                # Also add to a pandas dataframe
                df.append({
                    "Region": region,
                    "Composition": composition,
                    "Precision": f'{precision_avg * 100 :.2f}',
                    "Recall": f'{recall_avg * 100 :.2f}',
                    "F1": f'{f1_avg * 100 :.2f}',
                    "Accuracy": f'{accuracy_avg * 100 :.2f}',
                    "IoU": f'{iou_avg * 100 :.2f}'
                })

            df = pd.DataFrame(df)
            df = df.sort_values(by=["IoU"], ascending=False)
            print(df)
            df.to_excel(f"outputs/results_byhand_{region}.xlsx", index=False)
