# Enhanced Vegetation Index 2

def evi2(lsat8_raw):
    evi2 = 2.4 * (lsat8_raw["NIR"] - lsat8_raw["Red"]) / (lsat8_raw["NIR"] + lsat8_raw["Red"] + 1)

    return evi2