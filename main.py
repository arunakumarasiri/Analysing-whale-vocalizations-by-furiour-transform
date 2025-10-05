import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks, savgol_filter
from peak_analysis import analyze_and_plot_peaks
import pylab

pylab.rc('font', size=9)
fig = pylab.figure(figsize=(3.5, 7.5))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

def analyze_audio(file_path, label):

    signal, fs = librosa.load(file_path, sr=None)

    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft_values)

    positive_frequencies = frequencies[frequencies >= 0]
    positive_magnitude = magnitude[frequencies >= 0]
    
    return positive_frequencies / 10, positive_magnitude

def analyze_and_plot(ax, filename, ext, windowLength, polyOrder, xlimL, xlimH):

    file = f'{filename}.{ext}'
    frequencies, magnitude = analyze_audio(file, f"{filename} Sound")

    smoothed_magnitude = savgol_filter(magnitude, window_length=windowLength, polyorder=polyOrder)

    ax.plot(frequencies, magnitude, color='blue', alpha=0.2)
    ax.plot(frequencies, smoothed_magnitude, label=f"filtered signal", color='k', linewidth=1)
    ax.set_xlim(xlimL,xlimH)

    peaks_results  = analyze_and_plot_peaks(frequencies, smoothed_magnitude)

    for peak_x in peaks_results:
        ax.axvline(x=peak_x, color='red', linestyle='--', alpha=0.8, label=f'x={peak_x:.2f}', linewidth = 1)

    ax.legend()

pylab.rc('font', size=9)
fig = pylab.figure(figsize=(6.5, 8.5))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

analyze_and_plot(ax=ax1, filename='South_Pacific_Blue_whale_call', ext='wav', windowLength=201, polyOrder=3, xlimL=10, xlimH=30)
analyze_and_plot(ax=ax2, filename='Northeast_Pacific_blue_whale_call', ext='wav', windowLength=201, polyOrder=3, xlimL=10, xlimH=30)
analyze_and_plot(ax=ax3, filename='Western_Pacific_blue_whale_call', ext='wav', windowLength=201, polyOrder=3, xlimL=10, xlimH=30)
analyze_and_plot(ax=ax4, filename='Atlantic_blue_whale_call', ext='wav', windowLength=201, polyOrder=3, xlimL=10, xlimH=30)
analyze_and_plot(ax=ax5, filename='whale_like', ext='wav', windowLength=201, polyOrder=3, xlimL=20, xlimH=80)
analyze_and_plot(ax=ax6, filename='dolphin', ext='mp3', windowLength=201, polyOrder=3, xlimL=0, xlimH=350)

# analyze_and_plot(ax=ax6, filename='killer_whale', ext='mp3', windowLength=901, polyOrder=3, xlimL=450, xlimH=550)

ax1.set_title('South Pacific Blue whale')
ax2.set_title('Northeast Pacific blue whale')
ax3.set_title('Western Pacific blue whale')
ax4.set_title('Atlantic blue whale')
ax5.set_title('Unknown')
ax6.set_title('dolphin')

ax1.set_ylabel("Magnitude")
ax3.set_ylabel("Magnitude")
ax5.set_ylabel("Magnitude")
ax6.set_xlabel('Frequency(Hz)')
ax5.set_xlabel('Frequency(Hz)')
fig.set_tight_layout(True)

# plt.tight_layout() 
fig.savefig("frequency_spectrum.png", dpi = 300)
# plt.show()


# Whale sounds: https://whalesound.ca/whale-vocalizations/
# source: https://www.pmel.noaa.gov/acoustics/whales/sounds/sounds_whales_blue.html
# whales: https://archive.org/details/whale-songs-whale-sound-effects/killer-whale.mp3


################################################################################################################################

# Differentiating Whale Calls (Mating, Social, Alarm, Navigation)
# Mating Calls: Often repetitive, rhythmic, and low-frequency (e.g., humpback whale songs during breeding season).
# Social Calls: Shorter, varied in frequency and duration (e.g., orcas using dialects to communicate within pods).
# Alarm/Distress Calls: Sharp, high-pitched bursts, often indicating danger or separation.
# Echolocation Clicks: Used by toothed whales (e.g., sperm whales, dolphins) for navigation and hunting.