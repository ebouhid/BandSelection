# CRI550

def cri550(lsat8_raw):
    cri550 = (1/lsat8_raw["Blue"]) - (1/lsat8_raw["Green"])

    return cri550