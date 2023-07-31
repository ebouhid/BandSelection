import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import glob
import pandas as pd
from tqdm import tqdm


def evaluate_combination(combination, X_train, X_test, y_train, y_test):
    """
    Calculate the fitness of an combination by training an SVM classifier and
    evaluating its performance using accuracy.
    """
    # print(f'combination: {combination}')
    selected_bands = list(np.nonzero(combination)[0])

    X_train_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_train
    ]
    # print(f'X_train_sel[0].shape: {X_train_sel[0].shape}')
    X_test_sel = [
        segment[selected_bands, :, :].reshape(-1) for segment in X_test
    ]

    clf = SVC(C=100, gamma='scale', kernel='rbf', class_weight='balanced')
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    return balanced_accuracy_score(y_test, y_pred)


if __name__ == "__main__":
    # Loading dataset
    X_all = []
    y_all = []
    for path in glob.glob('data/dataset_v3-pca/forest/*'):
        X_all.append(np.load(path))
        y_all.append(0)

    for path in glob.glob('data/dataset_v3-pca/non_forest/*'):
        X_all.append(np.load(path))
        y_all.append(1)

    # perform split
    X_train, X_val, y_train, y_val = train_test_split(X_all,
                                                      y_all,
                                                      test_size=0.3,
                                                      random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val,
                                                    y_val,
                                                    test_size=0.5,
                                                    random_state=43)

    # Generate all possible combinations of 7-element arrays consisting of 0s and 1s
    combinations = np.array(list(np.ndindex((2, ) * 7)))

    results = []

    combinations = combinations[1:]  # remove all 0s combination
    loop = tqdm(combinations)
    for combination in loop:
        combination_string = ''.join([str(x) for x in combination])
        b_acc = evaluate_combination(combination, X_train, X_test, y_train,
                                     y_test)
        test_data = {
            'combination': combination_string,
            'Balanced Accuracy': b_acc
        }

        results.append(test_data)

    results_df = pd.DataFrame.from_records(results).sort_values(
        by='Balanced Accuracy', ascending=False)

    print(results_df)
    results_df.to_csv('results_bruteforce.csv', index=False)