import numpy as np
import rasterio
import glob
import os

if __name__ == '__main__':
    mask_paths = []
    aes = glob.glob('xingu/LS/AE*')

    for ae in aes:
        paths = glob.glob(os.path.join(ae, '*PRODES/'))
        for path in paths:
            for file in glob.glob(path + '*'):
                if file.split('_')[-1] == '8b.tif':
                    mask_paths.append(file)

    for mask in mask_paths:
        bands = []
        scope = mask.split('/')[-1].split('_')[1].lower()
        with rasterio.open(mask) as src:
            band = src.read(1)
            bands.append(band)

        prodes_data = np.dstack(bands)

        # Adaptation to consider only "forest" and "non-forest" classes
        prodes_data = np.where(prodes_data == 2, 0, 1)
        prodes_data = prodes_data.astype(np.uint8)

        np.save(f'truth_masks/truth_{scope}.npy', prodes_data)

    print('All done!')