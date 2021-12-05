import random

def changeContrast(images, power = 1.5, c= 0.7):
    """
    Change the contrast of the Spectrogram
    PARAMETERS
    ----------
    power: change the power
    c: bias
    
    RETURNS
    --------
    images: ndarray
    """
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)
    return images