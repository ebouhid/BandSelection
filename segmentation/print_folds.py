import numpy as np
NUM_FOLDS = 5

regions_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
folds = np.array_split(regions_list, NUM_FOLDS)

for fold in range(0, NUM_FOLDS):
        train_regions = np.concatenate(
            [regions for j, regions in enumerate(folds) if j != fold])
        test_regions = folds[fold]

        print(f'FOLD {fold}')
        print(f'Train: {train_regions}')
        print(f'Test: {test_regions}\n')