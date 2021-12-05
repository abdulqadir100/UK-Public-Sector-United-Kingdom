import scipy.io.wavfile as wav
import random
import math
import librosa
import numpy as np

def shiftTime(signal,sr=22050,time=None,direction=1):
    """
    PARAMETERS
    -----------
    signal: nd.array
    sr= Sample Rate
    Time: Time to Shift the Signal in milliseconds
    direction: Takes Value of 1 or 0
                if 0, Shift Signal to the left
                else Shift the Signal to the right
    RETURNS
    -------
    shift_signal
    
    """
    shift_max=random.randint(0,math.ceil((len(signal)/sr)/2))
    if time is None:
        shift=random.randint(0,(sr * shift_max))
        r_direction=random.randint(0,2)
        if r_direction == 1:
            shift=-shift
    else:
        shift=(time/1000) * sr
        shift=math.ceil(shift)
        if direction == 1:
            shift = - shift
    shift_signal=np.roll(signal,shift)
    
    if shift > 0:
        shift_signal[:shift]=0
    else:
        shift_signal[shift:]=0
    return shift_signal


def pitchShift(signal,sr=22050,n_steps=4,bins_per_octave=12):
    """
    PARAMETERS
    -----------
    signal: nd.array
    sr: Sample Rate
    n_steps: tones
    bins_per_octave
    
    RETURNS
    -------
    signal
    """
    return librosa.effects.pitch_shift(signal, sr, n_steps,bins_per_octave=bins_per_octave)

def timeStretch(signal,sr=22050,rate=2):
    """
    PARAMETERS
    ----------
    signal: nd.array
    sr: Sample Rate
    rate
    
    RETURNS
    -------
    new_signal
    
    """
    input_length=len(signal)
    data=librosa.effects.time_stretch(signal, rate)
    if len(data)>input_length:
        data = data
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
    return data