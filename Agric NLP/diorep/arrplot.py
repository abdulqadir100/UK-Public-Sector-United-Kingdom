import matplotlib.pyplot as plt
def plotSignal(signal,name_signal=""):
    """
    PARAMETERS
    -----------
    signal: Audio Signal
    name_signal: Name of the Audio Signal
    """
    plt.figure(figsize=(10,10))
    plt.plot(signal)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(f'PLOT OF {name_signal}')
    