# Alteration

def alteration(lsat8_raw):
    alteration = lsat8_raw["SWIR_1"] / lsat8_raw["SWIR_2"]
    return alteration