import scipy.io.wavfile as wav
import os
import numpy as np
from diorep.reduceNoise import retNoiseClip
import math
import random
import string

def get_amplitude(signal,percent=100):
    """
    PARAMETERS
    ----------
    signal: Audio Signal
    percent: Pecentile range
    RETURNS
    -------
    Amplitude
    """
    sr,arr=wav.read(signal)
    arr_max=np.percentile(arr,percent)
    return arr_max

def clip_category(value,noise_threshold,mid=50):
    """
    PARAMETERS
    ----------
    value: string
    noise_threshold: Maximum allowed noise threshold
    
    RETURNS
    -------
    Audio Category
    """
    if value < noise_threshold:
        return 'best'
    elif value > noise_threshold and value < mid:
        return 'good'
    else:
        return 'bad'

def gen_noise_file(signal,NOISE_DIR,noise_clip_length=500):
    """
    PARAMETERS
    ----------
    signal: Audio Signal
    NOISE_DIR: Path to save Generated Noise
    noise_clip_length: Length of Noise Clip
    
    RETURNS
    ------
    fname: Path to the Noise directory
    """
    if not os.path.exists(NOISE_DIR):
        os.makedirs(NOISE_DIR)
    fname=signal.split('/')
    fname=fname[1]
    fname=NOISE_DIR + '/' + fname
    sr,arr=wav.read(signal)
    arr_N=retNoiseClip(sr,arr,noise_clip_length)
    wav.write(fname,sr,arr_N)
    return fname

def assign_probability(value,prob_best=1,prob_good=0.7,prob_bad=0.2):
    """
    value: String
    prob_best: Probability Value
    prob_good: Probability Value
    prob_bad: Probability Value
    
    """
    if value == 'best':
        return prob_best
    elif value == 'good':
        return prob_good
    else:
        return prob_bad
    
def noise_clip(signal,noise_length):
    """
    PARAMETERS
    ----------
    signal: audio_clip
    noise_length: Length of Noise Clip in milliseconds
    
    RETURNS
    -------
    noise_clip
    """
    sr=22050
    len_=noise_length / 1000 * sr
    len_=int(len_)
    print(len_)
    return signal[:len_]

def generate_bkgnoise(df,noise_length=500,signal_length=60858,lower_dB_threshold=34,higher_dB_threshold=40):
    """
    PARAMETERS
    ----------
    df: DataFrame object (Train/Test)
    noise_length: Noise_Length from different Signals to Mix
    Lower_dB_threshold: Lower permissible noise threshold to mix
    Higher_dB_threshold: High permissible noise threshold to mix
    
    RETURNS
    -------
    noise: Returns an array of the Noise
    
    """
    noise_c=[]
    partition=(signal_length/22050)/(noise_length/1000)
    partition=math.ceil(partition)
    audio_loc_len=[]
    bool_=True
    count=0
    dup=[]
    while(bool_):
        #select Audio file randomly
        audio_loc=random.randint(0,len(df)-1)
        if df.dB_Noise_SPL.loc[audio_loc] in np.arange(lower_dB_threshold,higher_dB_threshold+1):
            if count == partition:
                bool_=False
            elif audio_loc in dup:
                pass
            else:
                count+=1
                dup.append(audio_loc)
                freq,arr=wav.read(df.fn.loc[audio_loc])
                noise=noise_clip(arr,noise_length)
                noise=noise_clip(arr,noise_length)
                noise_c.extend(noise)
    noise_c=np.array(noise_c)
    #noise_c[np.abs(noise_c) > 0.1]= float(decimal.Decimal(random.randrange(0, 10))/100)
    noise_c=noise_c[:signal_length]
    return noise_c


#Generate key

def random_key_gen(length=7,wn=False,stretch=False,shifting= False):
    """
    PARAMETERS
    ----------
    length: Length of Key
    wn: White Noise
    Shift: Time shifting
    Stretch: Stretch the Signal
    
    
    RETURNS
    -------
    returns Key
    
    
    
    """
    key = ''
    for i in range(length-2):
        key += random.choice( 
                             string.ascii_uppercase + 
                             string.digits+
                             string.ascii_uppercase
                            )
        
    if wn:
        return 'WN' + key
    elif stretch:
        return 'ST' + key
    elif shifting:
        return 'SH' + key
    else:
        return 'BK' + key
    
# Calculating the mean and Standard Deviation of Noise Clip



def silence_removal(signal,noise_length=200,sr=22050):
    
    """
    REFERENCE
    ----------
    International Journal of Innovative Research in Computer
    and Communication Engineering
    (An ISO 3297: 2007 Certified Organization)
    Vol. 4, Issue 4, April 2016
    Copyright to IJIRCCE DOI: 10.15680/IJIRCCE.2016. 0404046 6606
    Silence Removal from Audio Signal Using
    Framing and Windowing Method and Analyze
    Various Parameter
    PARAMETERS
    ----------
    signal: Audio Signal (nd.array)
    noise_length: Length of the noise signal in milliseconds
    """
    _,signal=wav.read(signal)
    arr=noise_clip(signal,noise_length)
    print(arr)
    mask=np.zeros_like(arr)
    mu=arr.mean()
    std=arr.std()
    signal=(signal-mu)/std
    mask = (signal > 3) * 1
    time_signal=len(signal)/sr
    partition = time_signal/10 # 10ms window
    partition=math.ceil(partition)
    part_len = 10/1000 * sr
    part_len = math.ceil(part_len)
    for i in range(partition):
        a=mask[part_len * i : part_len * (i+1)]
        if len(a==0) > len(a==1):
            mask[partition * i : partition * (i+1)] = 0
        else:
            mask[partition * i : partition * (i+1)] = 1
    new_signal = mask * signal
    new_signal = new_signal[new_signal!=0]
    
    return new_signal,mask


#Perform timeshift
def audio_time_shift(data,time=1600,dirpath='Time_shift'):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    keys=[]
    for index,signal in train.fn.items():
        freq,arr=wav.read(signal)
        data_n=np.roll(arr,time)
        key=random_key_gen(shifting=True)
        keys.append(key)
        wav.write('./'+dirpath+'/{}'.format(key)+'.wav',freq,data_n)
    return keys

#perform stretching
def stretch(data,rate=1,dirpath='Stretch_Audio'):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    keys=[]
    for index,signal in train.fn.items():
        freq,arr=wav.read(signal)
        data_n=librosa.effects.time_stretch(arr,rate)
        key=random_key_gen(stretch==True)
        keys.append(key)
        wav.write('./'+dirpath+'/{}'.format(key)+'.wav',freq,data_n)
    return keys
        