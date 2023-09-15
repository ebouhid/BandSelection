import random
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import logging

# Get command line arguments
exp_name = str(sys.argv[1])
dataset_path = str(sys.argv[2])

# Define seeds
seeds = [1, 10, 20, 30, 42]

logging.basicConfig(filename=f'results/logfile-{exp_name}.out',
                    level=logging.INFO)
logging.info(f'dataset_path: {dataset_path}')
logging.info(f'exp_name: {exp_name}')
for seed in seeds:
    # Set random seed
    np.random.seed(seed)

    # Starting log
    logging.info(f'seed: {seed}')

    # Loading dataset
    X_all = []
    y_all = []
    for path in glob.glob(dataset_path + '/forest/*.npy'):
        X_all.append(np.load(path).reshape(-1))
        y_all.append(0)

    for path in glob.glob(dataset_path + '/non_forest/*'):
        X_all.append(np.load(path).reshape(-1))
        y_all.append(1)

    # perform split
    X_train, X_val, y_train, y_val = train_test_split(X_all,
                                                      y_all,
                                                      test_size=0.3,
                                                      random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    test_size=0.5,
                                                    random_state=seed + 1)

    clf = SVC(C=100,
              gamma='scale',
              kernel='rbf',
              class_weight='balanced',
              random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Save results
    logging.info(f'Balanced Accuracy Score: {bal_acc}')
    logging.info('\n')

print('Done!')