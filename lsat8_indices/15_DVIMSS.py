# Differenced Vegetation Index MSS

def dvimss(lsat8_raw):
    dvimss = 2.4*lsat8_raw["NIR"] - lsat8_raw["Green"]

    return dvimss