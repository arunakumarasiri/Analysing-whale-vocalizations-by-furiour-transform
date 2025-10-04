import numpy as np
from scipy.signal import find_peaks, savgol_filter

def analyze_and_plot_peaks(x_values, y_values, smooth_window=21, poly_order=3, prominence_factor=0.2, min_distance=10):
    """
    Analyzes a dataset to:
    1. Find the highest peak.
    2. Calculate half of its magnitude as a threshold.
    3. Identify the top 3 highest peaks above that threshold using prominence filtering.
    4. Plot the dataset and draw vertical lines at the selected peaks.
    
    Parameters:
        x_values (array-like): X-axis values (e.g., frequencies).
        y_values (array-like): Y-axis values (e.g., magnitude).
        smooth_window (int): Window size for Savitzky-Golay smoothing.
        poly_order (int): Polynomial order for smoothing.
        prominence_factor (float): Fraction of max peak height to determine prominence threshold.
        min_distance (int): Minimum separation between peaks.

    Returns:
        peak_frequencies (list): The frequencies (x-values) of the significant peaks.
    """
    # Apply Savitzky-Golay filter for smoothing
    smoothed_y = savgol_filter(y_values, smooth_window, poly_order)

    # Find all peaks with prominence filtering
    peaks, properties = find_peaks(smoothed_y, prominence=prominence_factor * max(smoothed_y), distance=min_distance)

    # Get peak magnitudes and corresponding x values
    peak_magnitudes = smoothed_y[peaks]
    peak_x_values = x_values[peaks]

    # Identify the maximum peak
    if len(peak_magnitudes) == 0:
        print("No significant peaks found.")
        return []

    max_index = np.argmax(peak_magnitudes)
    max_peak_magnitude = peak_magnitudes[max_index]
    max_peak_x_value = peak_x_values[max_index]

    # Define threshold (half of the max peak's magnitude)
    threshold = max_peak_magnitude / 2

    # Filter peaks above the threshold
    significant_indices = np.where(peak_magnitudes > threshold)[0]

    # Sort these peaks by magnitude and take the top 3
    if len(significant_indices) > 1:
        sorted_indices = np.argsort(peak_magnitudes[significant_indices])[-3:]  # Get top 3 peaks
        selected_peaks_x = peak_x_values[significant_indices][sorted_indices]
    else:
        selected_peaks_x = [max_peak_x_value]  # If no other significant peaks, use only the highest one
    
    return selected_peaks_x