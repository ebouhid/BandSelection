from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import os
from skimage import io
from skimage.util import view_as_windows
import rasterio as r


def list_files(dir):
    filelist = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            # print(name)
            if (name.split('.')[-1] == 'png') or (name.split('.')[-1]
                                                  == 'npy'):
                filelist.append(os.path.join(root, name))

    filelist.sort()
    return filelist


"""
load_images_and_patchify: Carrega as cenas (img_dir) e suas respectivas verdades (mask_dir), gerando um dataset de acordo com os parâmetros:
- img_dir: local das cenas
- mask_dir: local das verdades correspondentes às cenas
- dataset_dir: diretorio onde o dataset será salvo; deve conter duas pastas "images" e "masks"
- patch_size: dimensão de cada patch (um quadrado de tamanho patch_size)
- stride: passo durante a formação dos patches
- clear_existing_db: limpa os diretorios dataset_dir/images e dataset_dir/masks, de forma que o dataset é formado do zero se True
"""


def load_images_and_patchify(img_dir,
                             mask_dir,
                             dataset_dir,
                             patch_size,
                             stride,
                             clear_existing_db=True):
    img_clear_path = './' + dataset_dir + '/images/*'
    msk_clear_path = './' + dataset_dir + '/masks/*'
    if clear_existing_db:
        os.system('rm ' + img_clear_path)
        os.system('rm ' + msk_clear_path)
    images = list_files(img_dir)
    masks = list_files(mask_dir)

    total_samples = 0

    print(f'Building image database')
    for file in tqdm(images):
        scope = (file.split('_')[-1]).replace(".npy", "")
        # img = io.imread(file)
        img = np.load(file)
        windows = view_as_windows(img, (patch_size, patch_size, img.shape[-1]),
                                  step=stride)
        # windows = windows.squeeze(axis=2)
        n_rows = windows.shape[0]
        n_cols = windows.shape[1]
        for i in range(n_rows):
            for j in range(n_cols):
                # img = Image.fromarray(windows[i, j, ...].squeeze())
                img = windows[i, j, ...].squeeze()
                np.save(f"{dataset_dir}/images/{scope}-{i}-{j}.npy", img)

    print(f'Building mask database')
    for file in tqdm(masks):
        scope = (file.split('_')[-1]).replace(".npy", "")
        # img = normalize_tif_mask(file, num_classes)
        msk = np.load(file)
        windows = view_as_windows(msk, (patch_size, patch_size, 1),
                                  step=stride)
        # windows = windows.squeeze(axis=2)
        n_rows = windows.shape[0]
        n_cols = windows.shape[1]

        total_samples += n_rows * n_cols

        for i in range(n_rows):
            for j in range(n_cols):
                # print(f'mask unique values: {np.unique(windows[i,j,...])}')
                msk = windows[i, j, ...].squeeze(0)
                np.save(f"{dataset_dir}/masks/{scope}-{i}-{j}.npy", msk)

    print(f'Total samples generated: {total_samples}')


# if __name__ == "__main__":
#     load_images_and_patchify('rgb_scenes', 'truth_masks',
#                              'xingu_dataset-v4.1-b4b6b7', 256, 256, True)
