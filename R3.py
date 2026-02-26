import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Import the necessary variables directly from the first script.
from R1 import x, Fs, N

# Generate an order-100 low-pass FIR filter with a cut-off frequency of 0.5.
h = signal.firwin(101, 0.5, pass_zero='lowpass')

# Filter the original signal x using the generated FIR filter.
xf = signal.lfilter(h, [1.0], x)

# Ensure that the data type of the filtered vector xf is the same as x.
xf = xf.astype(np.float32)

# Sample the filtered signal so that the new sampling rate is Fs divided by 2.
xf_sampled = xf[::2]

# Calculate the new sampling frequency.
Fs_xf = int(Fs / 2)

# Scale the float32 array to a 16-bit integer format for WAV compatibility.
xf_scaled = np.int16(xf_sampled * 32767)

# Save the downsampled filtered array as a wav file to listen to the result safely.
wavfile.write('filtered_downsampled_chirp.wav', Fs_xf, xf_scaled)

# Adjust the window length for the spectrogram to keep the window duration in seconds the same.
N_xf = int(N * (Fs_xf / Fs))

# Generate a Hanning window of size N_xf.
window_xf = signal.windows.hann(N_xf)

# Calculate the overlap to be exactly 3 times N_xf divided by 4.
noverlap_xf = int(3 * (N_xf / 4))

# Set the FFT length to be 4 times N_xf.
nfft_xf = 4 * N_xf

# Compute the spectrogram for the filtered and downsampled signal.
frequencies_xf, times_xf, Sxf = signal.spectrogram(xf_sampled, fs=Fs_xf, window=window_xf, nperseg=N_xf, noverlap=noverlap_xf, nfft=nfft_xf)

# Create a plot figure for the new spectrogram.
plt.figure(figsize=(10, 6))

# Plot the magnitude of the spectrogram in decibels.
plt.pcolormesh(times_xf, frequencies_xf, 10 * np.log10(Sxf + 1e-10), shading='gouraud')

# Add appropriate labels to the axes.
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (sec)')

# Add a descriptive title to the plot.
plt.title('Spectrogram of the Filtered and Downsampled Signal')

# Add a color bar to indicate the intensity in decibels.
plt.colorbar(label='Intensity (dB)')

# Display the generated plot on the screen.
plt.show()

# R3.a) Explain the differences in what is heard and observed, relative to what was heard and observed in item 2.
# When listening to the filtered and downsampled audio file, the pitch increases initially but then the sound smoothly fades into silence.
# In the spectrogram, the frequency curve rises until it hits the Nyquist limit of 2000 Hz and then abruptly disappears.
# Unlike the observation in item 2, no frequency components fold back down into the lower frequencies.
# This occurs because the anti-aliasing filter successfully removes the frequency components above 2000 Hz before the signal is sampled.
# The filter has a cut-off frequency of 0.5 in the discrete-time angular frequency scale, which corresponds exactly to 2000 Hz.
# Consequently, there is no high-frequency content left to cause aliasing, completely preventing the mirrored siren-like effect that was present in the previous signal.
# Keep in mind that this will result in information loss, although exclusively of high-frequency components that would have caused aliasing, instead
# of the entire signal being affected by the aliasing scrambling.