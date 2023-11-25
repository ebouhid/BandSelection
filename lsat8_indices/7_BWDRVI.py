# Blue-wide dynamic range vegetation index

def bwdrvi(lsat8_raw):
    bwdrvi = (0.1 * lsat8_raw["NIR"] - lsat8_raw["Blue"])\
        / (0.1 * lsat8_raw["NIR"] + lsat8_raw["Blue"])
    
    return bwdrvi