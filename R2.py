import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Import the necessary variables directly from the first script.
# Note that running this will execute the code in R1.py and display its plot.
from R1 import x, Fs, N

# Sample the original signal by taking every second element to obtain y(n) equal to x(2n)-
y = x[::2]

# Calculate the new sampling frequency.
# Since the signal is downsampled by a factor of 2, the new sampling frequency is half of the original.
Fs_y = int(Fs / 2)

# Scale the float32 array to a 16-bit integer format for compatibility with the WAV format.
y_scaled = np.int16(y * 32767)

# Save the new array as a wav file named 'aliased_chirp_signal.wav'.
# This allows one to listen to y(n) without PyAudio.
wavfile.write('aliased_chirp_signal.wav', Fs_y, y_scaled)

# R2.a) Indicate the sampling frequency of signal y(n) and justify the answer.
# The sampling frequency of signal y(n) is 4000 Hz.
# By sampling the original signal at every second index, the time step between consecutive samples is doubled.
# Consequently, the sampling rate is halved from the original 8000 Hz to 4000 Hz.

# Adjust the window length for the spectrogram to keep the window duration in seconds the same
# The original window length was 64 samples at 8000 Hz, which corresponds to 0.008 seconds.
# For the new sampling rate of 4000 Hz, the window length must be exactly 32 samples.
N_y = int(N * (Fs_y / Fs))

# Generate a Hanning window of size N_y.
window_y = signal.windows.hann(N_y)

# Calculate the overlap to be exactly 3 times N_y divided by 4.
noverlap_y = int(3 * (N_y / 4))

# Set the FFT length to be 4 times N_y.
nfft_y = 4 * N_y

# Compute the spectrogram for the downsampled signal.
frequencies_y, times_y, Syy = signal.spectrogram(y, fs=Fs_y, window=window_y, nperseg=N_y, noverlap=noverlap_y, nfft=nfft_y)

# Create a plot figure for the new spectrogram.
plt.figure(figsize=(10, 6))

# Plot the magnitude of the spectrogram in decibels.
# A tiny constant is added to the magnitude to prevent log of zero warnings.
plt.pcolormesh(times_y, frequencies_y, 10 * np.log10(Syy + 1e-10), shading='gouraud')

# Add appropriate labels to the axes.
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')

# Add a descriptive title to the plot.
plt.title('Spectrogram of the Downsampled Signal')

# Add a color bar to indicate the intensity in decibels.
plt.colorbar(label='Intensity (dB)')

# Display the generated plot on the screen.
plt.show()

# R2.b) Explain what is heard and observed.
# When listening to the new audio file, the pitch increases initially but then starts to decrease, creating a siren-like sweeping effect.
# In the spectrogram, the frequency curve rises until it hits a frequency of 2000 Hz.
# Once it exceeds this limit, the frequency component folds back down into the lower frequencies.
# This occurs because we are sampling a signal (x(n)), which has as maximum frequency of 4kHz (because it has no aliasing), 
# and our sampling frequency is lower than the Nyquist frequency of 8kHz (fs = 4kHz -> N = 32 for the Hanning window), which leads to aliasing.
# Think of it as the frequency components of the replica at -2pi and 2pi gradually folding back into the baseband as the original frequency increases, 
# creating a mirrored effect in the spectrogram and a corresponding change in the perceived pitch of the sound (the components of the replica at 2pi increase
# in frequency which leads to a decrease in the perceived frequency in the baseband).