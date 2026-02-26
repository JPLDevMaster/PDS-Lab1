import numpy as np
import matplotlib.pyplot as plt

# Load the signal data from the provided numpy file.
signal_data = np.load('sum_of_sines.npy')

# Print the shape of the loaded signal to confirm it has been loaded correctly.
print("Shape of the loaded signal:", signal_data.shape)

# Create a figure to plot the loaded time-domain signal.
plt.figure(figsize=(10, 6))

# Plot the signal against its sample indices since a specific sampling frequency was not provided for this file.
plt.plot(signal_data)

# Add appropriate labels to the axes.
plt.ylabel('Amplitude')
plt.xlabel('Sample Index')

# Add a descriptive title to the plot.
plt.title('Time-Domain Plot of the Corrupted Signal')

# Display the generated plot on the screen.
plt.show()

# R4.a) Can you identify the two frequencies in the signal?
# By simply observing the time-domain plot, it is extremely difficult to precisely identify the two frequencies.
# Plus, we don't know the sampling frequency, but we assume, for now, it is 1000Hz (just because it has 1000 samples...?).

# Compute the signal's frequency spectrum using the Fast Fourier Transform (FFT) algorithm.
spectrum = np.fft.fft(signal_data)

# Compute the magnitude spectrum.
magnitude_spectrum = np.abs(spectrum)

# Calculate the total number of samples.
N_samples = len(signal_data)

# Generate the normalized digital frequency axis.
# This maps the digital frequencies from 0 to 1 across the number of samples.
normalized_freqs = np.arange(N_samples) / N_samples

# Extract only the positive half of the spectrum for a clearer visualization.
half_N = N_samples // 2
positive_freqs = normalized_freqs[:half_N]
positive_magnitude = magnitude_spectrum[:half_N]

# Find the indices of the two highest peaks in the positive magnitude spectrum.
# The argsort function sorts the indices based on the magnitude values in ascending order, so the last two indices correspond to the largest peaks.
top_two_indices = np.argsort(positive_magnitude)[-2:]

# Extract the corresponding normalized digital frequencies using the found indices.
extracted_frequencies = positive_freqs[top_two_indices]

# Print the extracted frequencies to the console for verification.
print("The two dominant normalized digital frequencies (cycles/sample) are:", extracted_frequencies)

# Assuming a 1000 Hz sampling frequency, we can convert the normalized digital frequencies to actual frequencies in Hz.
# The conversion is done by multiplying the normalized frequency by the sampling frequency.
sampling_frequency = 1000
actual_frequencies = extracted_frequencies * sampling_frequency

print("The two dominant frequencies in Hz are:", actual_frequencies)

# Create a figure to plot the magnitude spectrum.
plt.figure(figsize=(10, 6))

# Plot the positive magnitude spectrum against the normalized digital frequency.
plt.plot(positive_freqs, positive_magnitude)

# Add appropriate labels to the axes.
plt.ylabel('Magnitude')
plt.xlabel('Normalized Digital Frequency (cycles/sample)')

# Add a descriptive title to the plot.
plt.title('Magnitude Spectrum of the Corrupted Signal')

# Display the generated plot on the screen to observe the peaks.
plt.show()

# R4.b) Comment on what is observed.
# Looking at the magnitude spectrum, two distinct and sharp peaks stand out above a noisy baseline.
# Based on the extracted data, the two dominant normalized digital frequencies are exactly 0.03 and 0.12 cycles/sample,
# which correspond to actual frequencies of 30 Hz and 120 Hz, respectively, assuming a sampling frequency of 1000 Hz.
# The noise floor remains far below these two distinct peaks.

# A copy of the original complex spectrum is created to preserve the original data.
filtered_spectrum = spectrum.copy()

# A threshold is defined to distinguish the dominant sinusoidal peaks from the noise floor.
# Given that we have the apriori knowledge that there are exactly two dominant frequencies, 
# we can iteratively increase the threshold until only two peaks remain in the positive magnitude spectrum.
# This leads to the optimal threshold that effectively eliminates the noise while preserving the two sine wave components.
max_magnitude = np.max(positive_magnitude)
threshold = 0.0
for th in np.arange(0.0, max_magnitude, 0.001):
    # The number of bins exceeding the current threshold is counted in the positive spectrum.
    peak_count = np.sum(positive_magnitude > th)
    
    # If exactly two peaks remain, the ideal threshold has been found.
    if peak_count == 2:
        threshold = np.round(th, 3)
        break

print(f"The determined threshold to isolate exactly two peaks is: {threshold:.3f}, which is {threshold/max_magnitude*100:.1f}% of the maximum magnitude.")

# All frequency bins with a magnitude below this threshold are set to exactly zero.
# This effectively eliminates the Gaussian white noise while preserving the two sine waves.
# Because this is applied to the full magnitude_spectrum, it correctly filters both the positive and negative frequency components.
filtered_spectrum[magnitude_spectrum < threshold] = 0

# The magnitude of the new filtered spectrum is computed for comparison.
filtered_magnitude_spectrum = np.abs(filtered_spectrum)

# Only the positive half of the filtered magnitude spectrum is extracted for plotting.
positive_filtered_magnitude = filtered_magnitude_spectrum[:half_N]

# A figure is created to compare the frequency spectrum of the original and the filtered signals.
plt.figure(figsize=(10, 6))

# The original noisy spectrum is plotted first with a slight transparency for context.
plt.plot(positive_freqs, positive_magnitude, label='Original Noisy Spectrum', alpha=0.5)

# The new filtered spectrum is plotted on top in a contrasting color.
plt.plot(positive_freqs, positive_filtered_magnitude, label='Filtered Spectrum', color='red')

# Plot everything.
plt.ylabel('Magnitude')
plt.xlabel('Normalized Digital Frequency (cycles/sample)')
plt.title('Comparison of Original and Filtered Spectra')
plt.legend()
plt.show()

# R4.c) Compare the frequency spectrum of the original and the filtered signals.
# By comparing the two spectra, it is visually clear that the noise floor has been completely flattened to zero.
# The only remaining components in the filtered spectrum are the two sharp peaks corresponding to the original sine waves.
# The frequency-domain thresholding successfully isolated the exact frequencies of interest from the broadband noise.