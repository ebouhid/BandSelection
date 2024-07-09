import os
import numpy as np


class UMDADataset:
    def __init__(self, segs_path, region_list, include_notanalyzed=False):
        self.segs_path = segs_path
        self.region_list = region_list
        self.include_notanalyzed = include_notanalyzed

        # Check if region list is valid
        for region in region_list:
            if region not in os.listdir(segs_path):
                raise ValueError(f"Region {region} not found in {segs_path}")

        self.X = []
        self.y = []

        # Load dataset
        for region in region_list:
            for segment in os.listdir(os.path.join(segs_path, region)):
                segpath = os.path.join(segs_path, region, segment)

                segclass = os.path.basename(segpath)
                segclass = segclass.split("_")[0]
                if segclass == "forest":
                    self.X.append(np.load(segpath))
                    self.y.append(0)
                elif segclass == "nonforest":
                    self.X.append(np.load(segpath))
                    self.y.append(1)
                else:
                    if self.include_notanalyzed:
                        self.X.append(np.load(segpath))
                        self.y.append(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_set(self):
        return self.X, self.y
