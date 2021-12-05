import numpy as np
import scipy.io.wavfile as wav
import random
from diorep.reduceNoise import reduceNoiseNR
import librosa
import cv2
from audiomentations import Compose, AddGaussianSNR,AddBackgroundNoise

def lenArray(signal):
    """
    PARAMETERS
    ----------
    signals: Dataframe column containing the signals
    RETURNS
    -------
    sample_length: Returns sample within the signal
                    type: ndarray
    """
    
    (frequency,array)=wav.read(signal)
    return len(array)


def padSample(len_signal,max_len,signal):
    """
    PARAMETERS
    ----------
    len_signal: len_signal
    len_max: Signal with the maximum sample
                type: integer
    signal: audio file
    RETURNS
    -------
    new_signal: Reconstructed signal
    
    """
    (frequency,array)=wav.read(signal)
    if len_signal < max_len:
        zero_concat=np.zeros(max_len-len(array))
        new_signal= np.concatenate((array,zero_concat),axis=0)
        #red_noise=reduceNoiseNR(frequency,new_signal,len_noise)
        return new_signal
    else:
        return array
    
def random_power(images, power = 1.5, c= 0.7,x=random.random()):
    """
    power: exponent to raise the random number generated
    c: bias for the exponent
    """
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(x*power + c)
    return images

def get_melspectr(signal,hp):
    # Create melspectrogram 
    spectr = librosa.feature.melspectrogram(signal, sr=hp.sr, n_mels=hp.n_mels, hop_length = hp.hop_length, fmin = 300)
    return spectr

def get_image(signal,hp,power=1.5,c=0.5,img_bias=80,x=random.random(),contrast=False):
    """
    PARAMETERS
    -----------
    signal: Audio Signal
    hp: Hyperparameters initialization
    power: exponent to raise the random number generated
    c: bias for the exponent
    
    RETURNS
    -------
    images
    """
    mel=get_melspectr(signal,hp)
    images = np.zeros((hp.n_mels, hp.img_width)).astype(np.float32)   
    # change the contrast
    if contrast:
        mel = random_power(mel, power = power, c= c,x=x)
    images = images + mel*(random.random() * hp.div_coef + 1)
    images = librosa.power_to_db(images.astype(np.float32), ref=np.max)
    images = (images+img_bias)/img_bias
    
    return images

def mono_to_color(X, eps=1e-6, mean=None, std=None):
    """
    PARAMETERS
    ----------
    x: image
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)

def resize(image, size=None):
    if size is not None:
        h, w, _ = image.shape
        new_w, new_h = int(w * size / h), size
        image = cv2.resize(image, (new_w, new_h))

    return image
        

def compute_db(max_signal,max_possible,bias=1e-7):
    """
    PARAMETERS
    ----------
    max_signal: Audio Signal
    max_possible: Max_possible over which other signals are compared
    bias: Avoid division by Zero
    RETURNS
    -------
    reference_signal: Maximum Amplitude to which other Signals are Compared
    SNR: Returns the dB
    """
    dB = 20 * np.log10((max_signal+bias)/max_possible)
    try:
        dB = int(dB)
    except OverflowError:
        dB=np.iinfo(int).max
    return dB
        
def get_wav_transforms(path):
    """
    PARAMETERS
    ----------
    path: Path to the Audio Signals
    
    RETURNS
    -------
    transform: Pytorch Transform
    """
    transforms = Compose(
        [
            AddGaussianSNR(max_SNR=0.5, p=0.5),
            AddBackgroundNoise(
                sounds_path=path, min_snr_in_db=0, max_snr_in_db=2, p=0.5
            ),
        ]
    )

    return transforms