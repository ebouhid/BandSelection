# Difference NIR/Green Green Difference Vegetation Index

def gdvi(lsat8_raw):
    gdvi = lsat8_raw["NIR"] - lsat8_raw["Green"]

    return gdvi