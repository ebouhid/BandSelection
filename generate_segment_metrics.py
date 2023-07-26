import numpy as np
from PIL import Image
from glob import glob
from skimage.measure import regionprops
import pandas as pd
from tqdm import tqdm

# 0 - Fundo
# 1 - Desmatamento recente
# 2 - Floresta
# 3 - Ñ analisado


def is_mixed(segment):
    # flattening segment
    segment = segment.flatten()

    NFP = np.count_nonzero(segment == 2)
    NP = np.count_nonzero(segment)  # desconsiderando o fundo (np.zeros)
    NNP = NP - NFP

    if NFP != 0 and NNP != 0:
        return True

    return False


def get_hor(segment):
    # flattening segment
    segment = segment.flatten()

    NFP = np.count_nonzero(segment == 2)
    NP = np.count_nonzero(segment)  # desconsiderando o fundo
    NNP = NP - NFP

    HoR = max([NFP, NNP]) / NP

    return HoR


def get_major_class(segment):
    if np.argmax(np.bincount(segment.flatten())) == 2:
        return "forest"
    elif np.argmax(np.bincount(segment.flatten())) == 1:
        return "non forest"
    elif np.argmax(np.bincount(segment.flatten())) == 3:
        return "not analyzed"
    else:
        return np.argmax(np.bincount(segment.flatten()))


compositions = ['pca', '467']

for composition in compositions:
    print(f'Now processing {composition}...')

    img_paths = sorted(glob(f'scenes_{composition}/*'))
    truth_paths = sorted(glob('truth_masks/*'))
    slic_paths = sorted(glob(f'slics/*{composition}.npy'))
    assert len(img_paths) == len(truth_paths) == len(slic_paths)

    all_segments = []
    for idx in range(len(img_paths)):
        image = np.asarray(Image.open(img_paths[idx]))
        truth = np.load(truth_paths[idx])
        slic = np.load(slic_paths[idx])

        props = regionprops(slic)
        segments_truth = []
        segments_image = []

        assert slic.shape[:2] == truth.shape[:2]

        loop = tqdm(enumerate(props),
                    desc=f'Region {idx + 1 :02d}',
                    total=len(props))

        for i, prop in loop:  # for each segment
            minr, minc, maxr, maxc = prop.bbox
            segment_truth = np.zeros((maxr - minr, maxc - minc))
            coords = np.array(prop.coords)
            for pixel in coords:
                segment_truth[pixel[0] - minr,
                              pixel[1] - minc] = truth[pixel[0], pixel[1]]
            hor = get_hor(segment_truth)
            classification = get_major_class(np.uint8(segment_truth))
            if classification in ['forest', 'non forest']:
                all_segments.append({
                    "Segment_label": prop.label,
                    "HoR": hor,
                    "Class": classification,
                    "Region": f'x{idx + 1 :02d}'
                })

    df = pd.DataFrame.from_records(all_segments)

    df.to_csv(f'segments_{composition}_all.csv', index=False)

print(f'All done!')