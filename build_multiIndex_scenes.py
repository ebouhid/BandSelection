import numpy as np
import spyndex
from glob import glob
from tqdm import tqdm
import warnings
import os

def get_region(path):
    return f"{path.split('/')[-1].split('.')[0].split('_')[-1]}"

def sublist(a, b):
    """Check if list a is a sublist of list b"""
    return len(set(a).intersection(set(b)))

if __name__ == "__main__":
    allbands_paths = sorted(glob('scenes_allbands/*'))
    print(f'Found {len(allbands_paths)} scenes')
    
    # Check for save path
    if not os.path.exists('scenes_multiIndex'):
        os.makedirs('scenes_multiIndex')
        print('Created scenes_multiIndex directory.')
    elif len(os.listdir('scenes_multiIndex')) != 0:
            # Remove previous files
            print('Removing previous files...')
            for file in os.listdir('scenes_multiIndex'):
                os.remove(f'scenes_multiIndex/{file}')

    # Load 7-band scenes
    scenes = [(np.load(path), path) for path in allbands_paths]
    print(f'Loaded {len(scenes)} scenes successfully')

    # Load Landsat-8 (Landsat-OLI) compatible spectral indexes
    platform = "Landsat-OLI"
    trouble_indices = ["AVI", "EVI", "GARI", "GEMI", "IBI", "NDDI", "NMDI", "SIPI", "TVI", "VARI", "VIBI", "IAVI"] # These indices were rasing warnings and yielding NaNs
    valid_indices = [index for index in spyndex.indices if platform in spyndex.indices[index].platforms \
                     and not sublist(['T1', 'T2', "PAR", "kNN", "kGG"], spyndex.indices[index].bands) \
                        and index not in trouble_indices]

    # Remove indexes that require thermal bands

    # Save a .txt file with the valid indices for the current platform
    i = np.load(allbands_paths[0]).shape[2]
    with open(f'valid_indices_{platform}.txt', 'w') as f:
        for index in valid_indices:
            f.write(f'{i}: {index}\n')
            i += 1

    print(f"Current platform is set as: {platform}.")
    print(f'Found {len(valid_indices)} valid indices for {platform}. List available at valid_indices_{platform}.txt')

    # Compute spectral indexes
    for scene, path in tqdm(scenes, desc='Computing spectral indexes'):
        scene = scene.transpose(2, 0, 1) / 255.
        scene = np.array(np.where(scene == 0., 1e-6, scene), dtype=np.float64)
        region = get_region(path)
        nans = np.count_nonzero(np.isnan(scene))
        # assert nans == 0, f'Found {nans} NaN values in {region} raw scene'
        params={
        "A":scene[0],
        "B":scene[1],
        "G":scene[2],
        "R":scene[3],
        "N":scene[4],
        "S1":scene[5],
        "S2":scene[6],
        "gamma": spyndex.constants['gamma'].default,
        "sla": spyndex.constants['sla'].default,
        "slb": spyndex.constants['slb'].default,
        "alpha": spyndex.constants['alpha'].default,
        "beta": spyndex.constants['beta'].default,
        "lambdaN": 0.851, # NIR wavelength for Landsat-8
        "lambdaR": 0.636, # Red wavelength for Landsat-8
        "lambdaG": 0.533, # Green wavelength for Landsat-8
        "epsilon": spyndex.constants['epsilon'].default, 
        "g": spyndex.constants['g'].default,
        "C1": spyndex.constants['C1'].default,
        "C2": spyndex.constants['C2'].default,
        "L": spyndex.constants['L'].default,
        "nexp": spyndex.constants['nexp'].default,
        "omega": spyndex.constants['omega'].default,
        "k": spyndex.constants['k'].default,
        "PAR": spyndex.constants['PAR'].default,
        "cexp": spyndex.constants['cexp'].default,
        "fdelta": spyndex.constants['fdelta'].default,
        }

        warnings.filterwarnings("error", category=RuntimeWarning) # uncomment to treat RuntimeWarnings as errors

        spectral_indexes = spyndex.computeIndex(valid_indices, params=params)
        scene_multiIndex = np.concatenate([scene, spectral_indexes], axis=0)

        nans = np.count_nonzero(np.isnan(scene_multiIndex))
        assert nans == 0, f'Found {nans} NaN values in {region} scene indexes'
        
        np.save(f'scenes_multiIndex/{region}.npy', scene_multiIndex)
        