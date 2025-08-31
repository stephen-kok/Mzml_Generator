import numpy as np
from scipy.signal import convolve, windows, find_peaks
from scipy import ndimage

# Placeholder for constants
DEFAULT_MZ_GRID_SIZE = 20000
DEFAULT_MASS_GRID_SIZE = 30000
DEFAULT_PEAK_SIGMA_MZ = 0.1
DEFAULT_CHARGE_SMOOTHING = 1.0
DEFAULT_MASS_SMOOTHING_DA = 10.0 # Sigma for mass smoothing in Daltons
DEFAULT_SOFTMAX_TEMP = 0.1
DEFAULT_PEAK_MIN_HEIGHT = 0.1
DEFAULT_PEAK_MIN_DISTANCE_DA = 100

def _create_peak_shape_kernel(mz_axis: np.ndarray, peak_sigma_mz: float, function: str = 'gaussian'):
    mz_step = mz_axis[1] - mz_axis[0]
    kernel_size = int(10 * peak_sigma_mz / mz_step) | 1
    kernel_mz = (np.arange(kernel_size) - kernel_size // 2) * mz_step
    if function == 'gaussian':
        sigma_points = peak_sigma_mz / mz_step
        kernel = windows.gaussian(kernel_size, sigma_points)
    else:
        gamma_points = (peak_sigma_mz / mz_step) / 2
        kernel = 1 / (1 + (kernel_mz / gamma_points)**2)
    return kernel / np.sum(kernel)

def _initialize_intensity_matrix(mz_axis_size: int, charge_axis_size: int):
    return np.ones((mz_axis_size, charge_axis_size))

def _softmax(x, T=1.0):
    e_x = np.exp((x - np.max(x)) / T)
    return e_x / e_x.sum()

def _find_peaks(mass_spectrum, mass_axis, min_height_fraction, min_distance_da):
    if np.max(mass_spectrum) == 0: return []
    min_height = min_height_fraction * np.max(mass_spectrum)
    min_distance_points = int(min_distance_da / (mass_axis[1] - mass_axis[0]))
    peak_indices, properties = find_peaks(mass_spectrum, height=min_height, distance=min_distance_points)
    return list(zip(mass_axis[peak_indices], properties["peak_heights"]))

def _calculate_uniscore(reconvolved_spectrum, measured_data):
    total_signal = np.sum(measured_data)
    if total_signal == 0: return 0.0
    residual = np.sum(np.abs(measured_data - reconvolved_spectrum))
    return 100.0 * max(0, 1 - (residual / total_signal))

def run_unidec(raw_spectrum: np.ndarray, config: dict):
    # 1. Extract parameters
    mz_range = config.get('mz_range')
    charge_range = config.get('charge_range')
    mass_range = config.get('mass_range')
    num_iterations = config.get('num_iterations', 100)
    adduct_mass = config.get('adduct_mass', 1.007825)
    peak_shape_function = config.get('peak_shape_function', 'gaussian')
    mz_grid_size = config.get('mz_grid_size', DEFAULT_MZ_GRID_SIZE)
    mass_grid_size = config.get('mass_grid_size', DEFAULT_MASS_GRID_SIZE)
    peak_sigma_mz = config.get('peak_sigma_mz', DEFAULT_PEAK_SIGMA_MZ)
    charge_smoothing_sigma = config.get('charge_smoothing_sigma', DEFAULT_CHARGE_SMOOTHING)
    softmax_temp = config.get('softmax_temp', DEFAULT_SOFTMAX_TEMP)
    peak_min_height = config.get('peak_min_height', DEFAULT_PEAK_MIN_HEIGHT)
    peak_min_distance_da = config.get('peak_min_distance_da', DEFAULT_PEAK_MIN_DISTANCE_DA)

    if any(v is None for v in [mz_range, charge_range, mass_range]):
        raise ValueError("mz_range, charge_range, and mass_range must be provided.")

    # 2. Create axes
    mz_axis = np.linspace(mz_range[0], mz_range[1], mz_grid_size)
    charge_axis = np.arange(charge_range[0], charge_range[1] + 1)
    mass_axis = np.linspace(mass_range[0], mass_range[1], mass_grid_size)
    mass_step = mass_axis[1] - mass_axis[0]

    # 3. Bin raw data
    measured_data_binned = np.zeros_like(mz_axis)
    indices = np.searchsorted(mz_axis, raw_spectrum[:, 0], side="left")
    valid_indices = (indices > 0) & (indices < mz_grid_size)
    if valid_indices.any():
        np.add.at(measured_data_binned, indices[valid_indices] - 1, raw_spectrum[valid_indices, 1])

    # 4. Initialize and create kernels
    intensity_matrix = _initialize_intensity_matrix(len(mz_axis), len(charge_axis))
    peak_kernel = _create_peak_shape_kernel(mz_axis, peak_sigma_mz, peak_shape_function)
    peak_kernel_flipped = np.flip(peak_kernel)

    if len(charge_axis) > 1:
        sigma_points_charge = charge_smoothing_sigma / (charge_axis[1] - charge_axis[0])
        charge_kernel_size = int(10 * sigma_points_charge) | 1
        charge_smoothing_kernel = windows.gaussian(charge_kernel_size, sigma_points_charge)
        charge_smoothing_kernel /= np.sum(charge_smoothing_kernel)

    # 5. Iterative Deconvolution Loop
    for _ in range(num_iterations):
        if len(charge_axis) > 1:
            intensity_matrix = ndimage.convolve1d(intensity_matrix, charge_smoothing_kernel, axis=1, mode='constant', cval=0.0)

        current_spectrum_estimate = np.sum(intensity_matrix, axis=1)
        reconvolved_spectrum = convolve(current_spectrum_estimate, peak_kernel, mode='same')
        reconvolved_spectrum[reconvolved_spectrum < 1e-9] = 1e-9

        error_ratio = measured_data_binned / reconvolved_spectrum
        correction_factor = convolve(error_ratio, peak_kernel_flipped, mode='same')

        intensity_matrix *= correction_factor[:, np.newaxis]

        current_sum = np.sum(intensity_matrix)
        if current_sum > 1e-9:
            norm_factor = np.sum(measured_data_binned) / current_sum
            intensity_matrix *= norm_factor

    # 6. Final Data Transformation and Processing
    mass_charge_matrix = np.zeros((len(mass_axis), len(charge_axis)))
    mass_grid = (mz_axis[:, np.newaxis] - adduct_mass) * charge_axis[np.newaxis, :]
    for i in range(len(charge_axis)):
        if np.any(intensity_matrix[:, i] > 0):
             mass_charge_matrix[:, i] = np.interp(mass_axis, mass_grid[:, i], intensity_matrix[:, i], left=0, right=0)

    mass_spectrum = np.sum(mass_charge_matrix, axis=1)
    mass_spectrum_processed = _softmax(mass_spectrum, T=softmax_temp)
    if np.max(mass_spectrum) > 0:
        mass_spectrum_processed *= np.max(mass_spectrum)

    detected_peaks = _find_peaks(mass_spectrum_processed, mass_axis, peak_min_height, peak_min_distance_da)
    final_estimate = np.sum(intensity_matrix, axis=1)
    final_reconvolved = convolve(final_estimate, peak_kernel, mode='same')
    uniscore = _calculate_uniscore(final_reconvolved, measured_data_binned)

    return detected_peaks, uniscore
