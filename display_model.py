#This file implements display models
import numpy as np


def RGBbt_709_np(input, peak_lum=110):
    black = peak_lum/1000.0
    lumin = (peak_lum-black)*(input/255.0)**2.2 + black
    return lumin



