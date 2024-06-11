from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import cv2

class XinguDataset(Dataset):
    def __init__(self,
                 scenes_dir,
                 masks_dir,
                 composition,
                 regions,
                 patch_size,
                 stride_size,
                 reflect_pad=False,
                 transforms=None):
        self.composition = composition
        self.patch_size = patch_size
        self.stride_size = stride_size

        self.img_path = scenes_dir
        self.msk_path = masks_dir

        self.image_paths = sorted(os.listdir(self.img_path))
        self.mask_paths = sorted(os.listdir(self.msk_path))

        self.images = []
        self.masks = []

        self.regions = regions
        self.transforms = transforms

        # load scenes
        for img_scene in self.image_paths:
            id = int((img_scene.split('_')[-1].split('.')[0])[1:])
            if id in regions:
                self.images.append(
                    np.load(os.path.join(self.img_path, img_scene)))

        for msk_scene in self.mask_paths:
            id = int((msk_scene.split('_')[-1].split('.')[0])[1:])
            if id in regions:
                self.masks.append(
                    np.load(os.path.join(self.msk_path, msk_scene)).squeeze())

        # patchify
        self.img_patches = []
        self.msk_patches = []

        for image in self.images:
            height, width, _ = image.shape
            for i in range(0, height, self.stride_size):
                for j in range(0, width, self.stride_size):
                    patch_image = image[i:i + self.patch_size,
                                        j:j + self.patch_size, :]
                    
                    # Dimension check; Pad if necessary
                    if patch_image.shape[0] != self.patch_size or patch_image.shape[1] != self.patch_size:
                        if reflect_pad:
                            rightpad = self.patch_size - patch_image.shape[1]
                            bottompad = self.patch_size - patch_image.shape[0]
                            patch_image = np.pad(patch_image, ((0, bottompad),(0, rightpad), (0,0)), mode='reflect')
                        else:
                            continue
                    self.img_patches.append(patch_image)

        for mask in self.masks:
            height, width = mask.shape
            for i in range(0, height, self.stride_size):
                for j in range(0, width, self.stride_size):
                    patch_mask = mask[i:i + self.patch_size,
                                      j:j + self.patch_size]
                    
                    # Dimension check; Pad if necessary
                    if patch_mask.shape[0] != self.patch_size or patch_mask.shape[1] != self.patch_size:
                        if reflect_pad:
                            rightpad = self.patch_size - patch_mask.shape[1]
                            bottompad = self.patch_size - patch_mask.shape[0]
                            patch_mask = np.pad(patch_mask, ((0, bottompad),(0, rightpad)), mode='reflect')
                        else:
                            continue
                    self.msk_patches.append(patch_mask)

        assert len(self.images) == len(self.masks)
        assert len(self.img_patches) == len(self.msk_patches)

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, idx):
        # get selected patches
        image = self.img_patches[idx]
        mask = self.msk_patches[idx]

        # create composition
        nbands = len(self.composition)
        combination = np.zeros(image.shape[:2] + (nbands,))

        for i in range(nbands):
            # Normalize this each band
            band  = image[:, :, (self.composition[i] - 1)]
            band = (band - np.min(band)) / (np.max(band) - np.min(band) + 1e-6)
            combination[:, :, i] = band

        mask = np.float32(mask)

        combination = combination.astype(np.float32)

        if self.transforms:
            aug = self.transforms(image=combination, mask=mask)
            combination, mask = aug['image'], aug['mask']
        
        combination = np.transpose(combination, (2, 0, 1))
        mask = np.where(mask == 2, 0, 1)
        mask = np.expand_dims(mask, axis=0)
        mask = np.float32(mask)
        return combination, mask
