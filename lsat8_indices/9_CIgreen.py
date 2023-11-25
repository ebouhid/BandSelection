# Clorophyll Index Green

def cigreen(lsat8_raw):
    cigreen = (lsat8_raw["NIR"] / lsat8_raw["Green"]) - 1
    return cigreen