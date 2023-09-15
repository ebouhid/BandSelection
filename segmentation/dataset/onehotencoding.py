import numpy as np

# * 1 = DESMATA
# * 2 = FLORESTA
# * 3 = NÃO ANALISA


class OneHotEncoding():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def perform(self, mask):
        shape = (self.num_classes, ) + mask.shape[:2]
        encoded = np.zeros(shape)
        for i in range(0, self.num_classes):
            encoded[i, :, :] = np.all(mask.reshape(-1, 1) == i,
                                        axis=1).reshape(shape[1:])

        return encoded

    def decode(self, encoded_mask):
        mask = np.argmax(encoded_mask, axis=0)
        mask += 1

        return mask
