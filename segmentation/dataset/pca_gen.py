from skimage import io
from sklearn.decomposition import PCA
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap
from tqdm import tqdm


def band_norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min) / (band_max - band_min))


loop = tqdm(range(1, 11))

for i in loop:
    band_paths = sorted(
        os.listdir(f'xingu/LS/AE-X{i :02d}/LS-AE-X{i :02d}_8b'))
    bands = []
    for fname in band_paths:
        with rasterio.open(
                os.path.join(f'xingu/LS/AE-X{i :02d}/LS-AE-X{i :02d}_8b',
                             fname)) as src:
            band = src.read(1)
            bands.append(band)

    landsat_data = np.dstack(bands)

    # Band 4 (Red)
    # Band 3 (Green)
    # Band 2 (Blue)
    true_rgb = np.zeros((landsat_data.shape[0], landsat_data.shape[1], 3))
    band_4 = landsat_data[:, :, 3]
    band_3 = landsat_data[:, :, 2]
    band_2 = landsat_data[:, :, 1]
    true_rgb[:, :, 0] = band_norm(band_4)
    true_rgb[:, :, 1] = band_norm(band_3)
    true_rgb[:, :, 2] = band_norm(band_2)
    true_rgb = np.uint8(true_rgb * 255)

    w, h, ch = landsat_data.shape
    landsat_data = np.reshape(landsat_data, (w * h, 7))
    landsat_data.shape

    pca = PCA(n_components=3)
    pc = pca.fit_transform(landsat_data)

    # rescaling
    pc_min = pc.min(axis=0)
    pc_max = pc.max(axis=0)
    pc_scaled = ((pc - pc_min) / (pc_max - pc_min)) * 255

    rgb = np.zeros((w, h, 3), dtype=np.uint8)
    rgb[:, :, 0] = pc_scaled[:, 0].reshape(w, h)
    rgb[:, :, 1] = pc_scaled[:, 1].reshape(w, h)
    rgb[:, :, 2] = pc_scaled[:, 2].reshape(w, h)

    np.save(f'pca_scenes/pca_x{i :02d}.npy', rgb)
