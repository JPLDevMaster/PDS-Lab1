import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Define the sampling frequency as 8000 samples per second.
Fs = 8000

# Create the time vector from 0 to 2 seconds.
t = np.arange(0, 2, 1/Fs)

# Set the constant k1 to 1000 as specified in the assignment.
k1 = 1000

# Implement the sampled signal.
# The formula calculates the cosine of 2 times pi times one third of k1 times t cubed.
x = np.cos(2 * np.pi * ((1/3) * k1 * t**3))

# Ensure that the data type of my vector x is float32.
x = x.astype(np.float32)

# Scale the float32 array to a 16-bit integer format for better compatibility with WAV.
x_scaled = np.int16(x * 32767)

# Save the array as a wav file named 'chirp_signal.wav'.
wavfile.write('chirp_signal.wav', Fs, x_scaled)

# R1.a) Comment on the relationship between what is heard and the signal that was created.
# When listening to the signal, one can hear a sound whose pitch increases rapidly over time.
# This makes perfect sense because the mathematical equation used defines a chirp whose instantaneous 
# frequency increases quadratically with time.

# R1.b) Spectrogram computation and display.
# Set the window size N to 64.
N = 64

# Generate a Hanning window of size N.
window = signal.windows.hann(N)

# Calculate the overlap to be exactly 3 times N divided by 4.
noverlap = int(3 * (N / 4))

# Setthe FFT length to be 4 times N.
nfft = 4 * N

# Compute the spectrogram using the parameters defined above.
frequencies, times, Sxx = signal.spectrogram(x, fs=Fs, window=window, nperseg=N, noverlap=noverlap, nfft=nfft)

# Create a plot figure for the spectrogram.
plt.figure(figsize=(10, 6))

# Plot the magnitude of the spectrogram in decibels using a colormesh.
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')

# Add appropriate labels to the axes.
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')

# Add a descriptive title to the plot.
plt.title('Spectrogram of the Sampled Chirp Signal')

# Add a color bar to indicate the intensity in decibels.
plt.colorbar(label='Intensity (dB)')

# Display the generated plot on the screen.
plt.show()

# R1.c) Comment on the relationship between the spectrogram and the sound heard.
# By looking at the spectrogram, one can visually track a distinct curve that bends upwards toward higher frequencies.
# This upward visual curve perfectly matches the increasingly high-pitched sweeping sound that is heard when one playes the audio.

# Because the sampling frequency is 8000Hz and maximum signal frequency is 1000 * 2^2 = 4kHz <= 8/2kHz = 4kHz, there is no aliasing in this case.
# The Nyquist frequency is 4kHz * 2 = 8kHz (take the derivative of the cosine argument).