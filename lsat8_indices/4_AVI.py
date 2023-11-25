# Ashburn Vegetation Index

def avi(lsat8_raw):
    avi = 2 * lsat8_raw["NIR"] - lsat8_raw["Red"]
    return avi