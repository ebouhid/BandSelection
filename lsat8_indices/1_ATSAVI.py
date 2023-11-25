# Adjusted transformed soil-adjusted VI

# Landsat8 Band dictionary
lsat8_bands = {
    "Coastal": 0,
    "Blue": 1,
    "Green": 2,
    "Red": 3,
    "NIR": 4,
    "SWIR_1": 5,
    "SWIR_2": 6,
}

def ATSAVI(lsat8_raw):
    atsavi = 1.22 * (lsat8_raw["NIR"] - 1.22 * lsat8_raw["Red"] - 0.03) / (1.22 * lsat8_raw["NIR"] + lsat8_raw["Red"] - 1.22*0.03 + 0.08 * (1 + 1.22**2))
    return atsavi
