import numpy as np
import noisereduce as nr
import math

def reduceNoiseNR(freq,arr,req_length):
    """
    PARAMETERS
    -----------
    freq: Sampling frequency of thea audio signal
    arr: Numpy array of the signal
    req_length: Length of noise signal required in milli-seconds

    RETURNS
    --------
    reduce_signal: Array

    """
    noise_clip=retNoiseClip(freq,arr,req_length)
    reduce_signal=nr.reduce_noise(arr,noise_clip)
    return reduce_signal


def retNoiseClip(freq,arr,req_length):
    """
    PARAMETERS
    -----------
    freq: Sampling frequency of thea audio signal
    arr: Numpy array of the signal
    req_length: Length of noise signal required in milli-seconds

    RETURNS
    --------
    noise_clip: Array
    """
    length_noise_clip=math.ceil(req_length/1000 * freq)
    noise_clip=arr[:length_noise_clip]
    return noise_clip



    
