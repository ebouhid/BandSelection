import os

# Set environment variables to disable multithreading in numpy/mahotas/skimage
# to prevent thread explosion when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from skimage.measure import regionprops
import mahotas
from tqdm import tqdm
import sys
import multiprocessing
import argparse
import skimage.io as io


def load_superpixels(seg_path):
    if seg_path.endswith(".npy"):
        return np.load(seg_path)
    elif seg_path.endswith((".png", ".pgm")):
        return io.imread(seg_path)
    else:
        raise ValueError("Unsupported file format")

def get_hor(segment):
    # flattening segment
    segment = segment.flatten()

    NFP = np.where(segment == 2, 1, 0).sum()
    NP = segment.size
    NNP = NP - NFP

    HoR = max([NFP, NNP]) / NP

    return HoR


def get_major_class(segment):
    if np.argmax(np.bincount(segment.flatten())) == 2:
        return "forest"
    elif np.argmax(np.bincount(segment.flatten())) == 3:
        return "notanalyzed"
    else:
        return "nonforest"

def get_region(path):
    return f"{path.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0]}"


def evaluate_segment(segment):
    classification = get_major_class(segment)

    if (segment.shape[0] * segment.shape[1] > 70) and (get_hor(segment) > 0.7):
        return True

    return False

def process_segment(data):
    region, label, segment_image, segment_truth, dataset_name = data
    segment_class = get_major_class(segment_truth)
    segment_id = label

    if evaluate_segment(segment_truth):
        segment_haralick = [mahotas.features.haralick(segment_image[:, :, channel]) for channel in range(segment_image.shape[2])]
        np.save(f'data/classification_datasets/{dataset_name}/{region}/{segment_class}_{segment_id}.npy', segment_haralick)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, default=16)
    parser.add_argument("--dataset_name", "-d", type=str, required=True)
    parser.add_argument("--scenes_path", "-s", type=str, required=True)
    parser.add_argument("--truth_path", "-t", type=str, required=True)
    parser.add_argument("--slic_path", "-l", type=str, required=True)
    
    args = parser.parse_args()

    # Cap processes to avoid overloading system
    cpu_count = multiprocessing.cpu_count()
    num_processes = min(args.processes, cpu_count - 2)  # Leave 2 CPUs for system
    print(f"Using {num_processes} processes (available CPUs: {cpu_count})")

    regions = [f"x{x:02d}" for x in range(1, 11) if x != 5]

    # Create directories
    for region in regions: 
        os.makedirs(f'data/classification_datasets/{args.dataset_name}/{region}/', exist_ok=True)

    # Process each region as a batch to limit memory usage
    with multiprocessing.Pool(num_processes) as pool:
        for region in regions:
            image_path = f'{args.scenes_path}/{region}.npy'
            image = np.load(image_path).astype(np.uint8)

            truth_path = f'{args.truth_path}/truth_{region}.npy'
            truth = np.load(truth_path).astype(np.uint8)

            slic_path = f'{args.slic_path}/slic_{region}.npy'
            slic = load_superpixels(slic_path)

            assert truth.shape[:2] == slic.shape[:2], f"Truth shape: {truth.shape} | Slic shape: {slic.shape}"
            assert truth.shape[:2] == image.shape[:2], f"Truth shape: {truth.shape} | Image shape: {image.shape}"

            props = regionprops(slic)

            # Pre-extract segment crops for this region only
            segments = []
            for prop in props:
                minr, minc, maxr, maxc = prop.bbox
                segment_image = image[minr:maxr, minc:maxc, :].copy()
                segment_truth = truth[minr:maxr, minc:maxc].copy()
                segments.append((region, prop.label, segment_image, segment_truth, args.dataset_name))

            # Process this region's segments
            chunksize = max(1, len(segments) // (10 * num_processes))
            list(tqdm(pool.imap(process_segment, segments, chunksize=chunksize), 
                      total=len(segments), desc=f'Processing {region}'))
            
            # Free memory before next region
            del image, truth, slic, segments

