# Enhanced Vegetation Index

def evi(lsat8_raw):
    evi = 2.5 * (lsat8_raw["NIR"] - lsat8_raw["Red"])\
        / (lsat8_raw["NIR"] + 6 * lsat8_raw["Red"] - 7.5 * lsat8_raw["Blue"] + 1)
    
    return evi