# Chlorophyll Vegetation Index

def cvi(lsat8_raw):
    cvi = lsat8_raw["NIR"] * lsat8_raw["Red"] / (lsat8_raw["Green"]**2)
    return cvi