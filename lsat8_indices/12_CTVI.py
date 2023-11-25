# Correted Transformed Vegetation Index

from math import sqrt, abs

def ctvi(lsat8_raw):
    quot = ((lsat8_raw["Red"] - lsat8_raw["Green"]) / (lsat8_raw["Red"] + lsat8_raw["Green"]) + 0.5)\
     / abs((lsat8_raw["Red"] - lsat8_raw["Green"]) / (lsat8_raw["Red"] + lsat8_raw["Green"]) + 0.5)
    
    root = sqrt(((lsat8_raw["Red"] - lsat8_raw["Green"]) / (lsat8_raw["Red"] - lsat8_raw["Green"])) + 0.5)

    ctvi = quot * root

    return ctvi