from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from PIL import Image
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA for satellite scenes')
    parser.add_argument('--input-dir', '-i', type=str, help='Directory containing the scenes')
    parser.add_argument('--output-dir', '-o', type=str, help='Directory to save the PCA images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for path in (os.listdir(args.input_dir)):
        region = path.split('_')[-1].replace('.npy', '')
        print(f'Now processing {region}')
        multispectral_data = np.load(os.path.join(args.input_dir, path))
        w, h, ch = multispectral_data.shape
        multispectral_data = np.reshape(multispectral_data, (w * h, ch))
        print(f'unique before scaling: {np.unique(multispectral_data)}')
        zero_one_scaler = MinMaxScaler()
        multispectral_data = zero_one_scaler.fit_transform(multispectral_data)
        print(f'unique after scaling: {np.unique(multispectral_data)}')

        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(multispectral_data)

        scaler = MinMaxScaler(feature_range=(0, 255))
        principal_components = scaler.fit_transform(principal_components)

        print(f'{region} | {np.unique(principal_components)}')

        falsecolor_pca = np.zeros((w, h, 3), dtype=np.uint8)
        falsecolor_pca[:, :, 0] = principal_components[:, 0].reshape(w, h)
        falsecolor_pca[:, :, 1] = principal_components[:, 1].reshape(w, h)
        falsecolor_pca[:, :, 2] = principal_components[:, 2].reshape(w, h)

        np.save(f'{args.output_dir}/pca_{region}.npy', falsecolor_pca)
        falsecolor_pca = Image.fromarray(falsecolor_pca)
        falsecolor_pca.save(f'{args.output_dir}/pca_{region}.png')
