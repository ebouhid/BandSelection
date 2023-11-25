# Atmospherically Resistant Vegetation Index 2

def arvi2(lsat8_raw):
    arvi2 = -0.18 + 1.17 * (lsat8_raw["NIR"] - lsat8_raw["Red"]) / (lsat8_raw["NIR"] + lsat8_raw["Red"])
    return arvi2