import numpy as np
from skimage.measure import regionprops
import mahotas
import os
from tqdm import tqdm
import sys


def get_hor(segment):
    # flattening segment
    segment = segment.flatten()


    NFP = np.count_nonzero(segment == 2)
    NP = np.count_nonzero(segment) # not considering background
    NNP = NP - NFP

    HoR = max([NFP, NNP]) / NP

    return HoR

def normalize_band(band):
    # Normalize band to uint8
    band = band - np.min(band)
    band = band / (np.max(band)) * 255
    band = band.astype(np.uint8)

    return band

def get_major_class(segment):
    segment = np.array(segment, dtype=np.uint8)
    if np.argmax(np.bincount(segment.flatten())) == 2:
        return "forest"
    elif np.argmax(np.bincount(segment.flatten())) == 1:
        return "non_forest"
    elif np.argmax(np.bincount(segment.flatten())) == 3:
        return "not analyzed"
    else:
        return np.argmax(np.bincount(segment.flatten()))
    
def evaluate_segment(segment):
    classification = get_major_class(segment)

    if (segment.shape[0] * segment.shape[1] > 70) and (get_hor(segment > 0.7))\
        and (classification in ["forest", "non_forest"]):
        return True

    return False

if __name__ == "__main__":
    # Get region from command line
    region = sys.argv[1]

    val_region = ['x08']
    test_regions = ['x01', 'x03']

    scope = 'test' if region in test_regions else 'val' if region in val_region else 'train'

    # Create directories
    os.makedirs(f'data/classification_dataset/{scope}/forest', exist_ok=True)
    os.makedirs(f'data/classification_dataset/{scope}/non_forest', exist_ok=True)

    image_path = f'scenes_sentinel_ndvi/{region}.npy'
    image = np.load(image_path)
    for i in range(image.shape[0]):
        image[i, :, :] = normalize_band(image[i, :, :])
    
    image = np.array(image, dtype=np.uint8)

    truth_path = f'truth_masks_sentinel/truth_{region}.npy'
    truth = np.load(truth_path)

    slic_path = f'slics_sentinel/mask_slic_{region}.npy'
    slic = np.load(slic_path)

    assert truth.shape[:2] == slic.shape[:2]
    assert truth.shape[:2] == image.shape[:2]

    props = regionprops(slic)

    segments = []

    for prop in tqdm(props, desc=f'Processing {region}'):
        minr, minc, maxr, maxc = prop.bbox
        segment_image = image[minr:maxr, minc:maxc, :]
        segment_truth = truth[minr:maxr, minc:maxc]
        segment_class = get_major_class(segment_truth)
        segment_id = prop.label                

        if evaluate_segment(segment_truth):
            # Computing Haralick features for the segment
            segment_haralick = [mahotas.features.haralick(segment_image[:, :, channel]) for channel in range(segment_image.shape[2])]

            # Saving haralick segment
            np.save(f'data/classification_dataset/{scope}/{segment_class}/{region}_{segment_id}.npy', segment_haralick)
