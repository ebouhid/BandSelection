# Coloration Index

def ci(lsat8_raw):
    ci = (lsat8_raw["Red"] - lsat8_raw["Blue"]) / lsat8_raw["Red"]
    return ci