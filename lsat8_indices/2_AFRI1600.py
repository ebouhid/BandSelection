# Aerosol-Free Vegetation Index 1600

def afri1600(lsat8_raw):
    afri1600 = lsat8_raw["NIR"] - 0.66 * (lsat8_raw["SWIR_1"] / (lsat8_raw["NIR"] + 0.66 * lsat8_raw["SWIR_1"]))
    return afri1600