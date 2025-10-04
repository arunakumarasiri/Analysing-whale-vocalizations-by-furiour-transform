import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the WAV file (preserving original sampling rate)
file_path = "whale_like.wav"  # Replace with your file path
signal, fs = librosa.load(file_path, sr=None)

# Remove DC offset
signal = signal - np.mean(signal)

# Apply a window function (Hamming)
window = np.hamming(len(signal))
signal = signal * window

# Increase FFT length for better resolution
N = 2**14  # Try 16384 or higher for finer resolution
fft_values = np.fft.fft(signal, n=N)
frequencies = np.fft.fftfreq(N, d=1/fs)
magnitude = np.abs(fft_values)

# Keep only the positive frequencies
half_N = len(frequencies) // 2
frequencies = frequencies[:half_N]
magnitude = magnitude[:half_N]

# Plot the frequency spectrum
plt.figure(figsize=(10, 5))
plt.plot(frequencies, magnitude, color='b')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.xlim(0, fs / 2)  # Show up to Nyquist frequency (fs/2)
plt.grid()
plt.show()
