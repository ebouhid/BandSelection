class CrossValidationIterator:
    def __init__(self, regions):
        self.regions = regions
        self.num_regions = len(regions)
        self.current_fold = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fold >= self.num_regions:
            raise StopIteration
        else:
            test_region = self.regions[self.current_fold]
            train_regions = [r for r in self.regions if r != test_region]
            self.current_fold += 1
            return train_regions, [test_region]



if __name__ == "__main__":
    # regions = ["x01", "x02", "x03", "x04", "x06", "x07", "x08", "x09"]
    regions = [1, 2, 3, 4, 6, 7, 8, 9, 10]  # Regions from 1 to 10 (excluding 5)
    cv_iterator = CrossValidationIterator(regions)
    for fold, (train_regions, test_regions) in enumerate(cv_iterator):
        print(f"Fold {fold}")
        print(f"Train index: {train_regions}")
        print(f"Test index: {test_regions}")
        print()

