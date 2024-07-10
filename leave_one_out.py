import umda as umda
from umda_dataset import UMDADataset
import argparse
import pandas as pd
import os
import logging


class LeaveOneOutIterator:
    def __init__(self, region_list):
        self.region_list = region_list
        self.num_regions = len(region_list)

    def __iter__(self):
        self.current_region = 0
        return self

    def __next__(self):
        train_regions = [self.region_list[i] for i in range(
            self.num_regions) if i != self.current_region]
        test_region = self.region_list[self.current_region]

        self.current_region += 1

        return train_regions, test_region


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs_path", type=str, required=True)
    parser.add_argument("--include_notanalyzed", type=bool, default=False)
    parser.add_argument("--population_size", type=int, default=60)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--num_parents", type=int, default=15)
    parser.add_argument("--num_offspring", type=int, default=45)
    parser.add_argument("--inf_lim", type=float, default=0.125)
    parser.add_argument("--sup_lim", type=float, default=0.875)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    all_fold_results = []
    os.makedirs(args.output_dir, exist_ok=True)

    regions = [f"x{x:02d}" for x in range(1, 11) if x != 5]
    loop = LeaveOneOutIterator(regions)

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    for fold_num, (train_regions, test_region) in enumerate(loop):
        log_filename = f"{args.output_dir}/log_fold_{fold_num}.log"
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))

        # Clear existing handlers, and add the new handler
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.addHandler(file_handler)

        logging.info(f"Starting fold {fold_num}")
        logging.info(f"Train regions: {train_regions}")
        logging.info(f"Test region: {[test_region]}")

        logging.info(f"Seed: {args.seed}")

        # Load datasets
        train_dataset = UMDADataset(
            args.segs_path, train_regions, args.include_notanalyzed)
        test_dataset = UMDADataset(
            args.segs_path, [test_region], args.include_notanalyzed)

        X_train, y_train = train_dataset.get_set()
        X_test, y_test = test_dataset.get_set()

        # Call UMDA
        fold_results = umda.umda(X_train, X_test, y_train, y_test, args.population_size,
                                 args.num_generations, args.num_parents, args.num_offspring,
                                 args.inf_lim, args.sup_lim, seed=args.seed)
        fold_results['fold'] = fold_num
        all_fold_results.append(fold_results)
        fold_results.to_csv(
            f"{args.output_dir}/fold_{fold_num}.csv", index=False)
        logging.info(f"Completed fold {fold_num}")
        print(f"Completed fold {fold_num}")

    all_fold_results = pd.concat(all_fold_results)
    all_fold_results.to_csv(f"{args.output_dir}/all_folds.csv", index=False)
