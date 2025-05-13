from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal

# Define paths
INPUT_PATH = r"C:/Users/WILLIAM7771/Desktop/procc/predicting-driver-stress-using-deep-learning/data/preprocessed_data"
OUTPUT_PATH = Path(INPUT_PATH).joinpath("final_data")

# Sampling frequency
SAMPLING_FREQUENCY = 15.5

# Frequency bands for respiration features
FREQ_BANDS = {
    "ulf": (0.0, 0.1),
    "vlf": (0.1, 0.2),
    "lf": (0.2, 0.3),
    "hf": (0.3, 0.4),
}

# Functions to calculate features
def create_resp_features(resp, band):
    """
    Calculates Welch Average Modified Periodogram with Hanning Window.
    """
    frequencies, powers = signal.welch(
        x=resp,
        fs=SAMPLING_FREQUENCY,
        window="hann",
        nperseg=len(resp),
        noverlap=len(resp) // 2,
    )
    return np.sum(powers[(frequencies >= FREQ_BANDS[band][0]) & (frequencies <= FREQ_BANDS[band][1])])

def create_gsr_features(gsr, return_type):
    """
    Calculates Peaks, Magnitude, Duration, and Area of GSR signal.
    """
    peaks, _ = signal.find_peaks(gsr)
    widths, heights, left_ips, _ = signal.peak_widths(gsr, peaks, rel_height=1)

    if return_type == "frequency":
        return len(peaks)
    elif return_type == "magnitude":
        return np.sum(gsr[peaks] - heights)
    elif return_type == "duration":
        return np.sum(peaks - left_ips)
    elif return_type == "area":
        return np.sum(0.5 * (gsr[peaks] - heights) * (peaks - left_ips))

def create_hrv_feature(hr):
    """
    Calculates Lomb-Scargle periodogram for HRV analysis.
    """
    periods = np.linspace(0.01, 0.5, 50)
    angular_frequencies = (2 * np.pi) / periods
    timestamps = np.linspace(1 / SAMPLING_FREQUENCY, len(hr) * (1 / SAMPLING_FREQUENCY), num=len(hr))

    try:
        lomb = signal.lombscargle(timestamps, hr, angular_frequencies, normalize=True)
        low_freq_power = np.sum(lomb[:8])  # 0-0.08 Hz
        high_freq_power = np.sum(lomb[14:])  # 0.15-0.5 Hz
        return low_freq_power / high_freq_power
    except ZeroDivisionError:
        print("Failed to calculate HRV ratio. Returning mean instead.")
        return np.mean(hr)

def filter_minutes(x, start_minute, end_minute):
    """
    Filters data based on the specified time range.
    Returns True if any row in the group falls within the time range, otherwise False.
    """
    cumulative_count = pd.Series(np.arange(len(x)), index=x.index)
    maximum = cumulative_count.max()
    minimum = cumulative_count.min()

    if maximum - minimum < end_minute - start_minute:
        return True  # Keep the group if it's too short to filter
    return ((cumulative_count >= start_minute) & (cumulative_count <= end_minute)).any()

# Dictionary of functions to apply to downsampled data for the pandas group by method.
functions_to_apply = {
    "EMG": np.mean,
    "footGSR": [
        np.mean,
        np.std,
        lambda x: create_gsr_features(gsr=x, return_type="frequency"),
        lambda x: create_gsr_features(gsr=x, return_type="magnitude"),
        lambda x: create_gsr_features(gsr=x, return_type="duration"),
        lambda x: create_gsr_features(gsr=x, return_type="area"),
    ],
    "handGSR": [
        np.mean,
        np.std,
        lambda x: create_gsr_features(gsr=x, return_type="frequency"),
        lambda x: create_gsr_features(gsr=x, return_type="magnitude"),
        lambda x: create_gsr_features(gsr=x, return_type="duration"),
        lambda x: create_gsr_features(gsr=x, return_type="area"),
    ],
    "HR": [np.mean, np.std, create_hrv_feature],
    "RESP": [
        np.mean,
        np.std,
        lambda x: create_resp_features(resp=x, band="ulf"),
        lambda x: create_resp_features(resp=x, band="vlf"),
        lambda x: create_resp_features(resp=x, band="lf"),
        lambda x: create_resp_features(resp=x, band="hf"),
    ],
    "Stress": np.mean,
}

# Final column names for the final dataset.
column_names = [
    "time",
    "EMG_mean",
    "footGSR_mean", "footGSR_std", "footGSR_frequency", "footGSR_magnitude", "footGSR_duration", "footGSR_area",
    "handGSR_mean", "handGSR_std", "handGSR_frequency", "handGSR_magnitude", "handGSR_duration", "handGSR_area",
    "HR_mean", "HR_std", "HRV_ratio",
    "RESP_mean", "RESP_std", "RESP_ulf", "RESP_vlf", "RESP_lf", "RESP_hf",
    "Stress_mean",
]

def process_single_file(file_path, output_dir, downsample_frequency="10S", time_range=(3, 9)):
    """
    Processes a single file and saves the results.
    """
    file_name = Path(file_path).stem
    print(f"\nProcessing file: {file_name}")

    # Load data
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Failed to load file: {file_name}. Error: {e}")
        return None

    # Clean column names
    data.columns = [col.split("-")[0].replace(" ", "") for col in data.columns]
    print(f"Columns in {file_name}: {data.columns.tolist()}")

    # Check for required columns
    required_columns = {"EMG", "footGSR", "handGSR", "HR", "RESP", "Stress"}
    if not required_columns.issubset(data.columns):
        missing_columns = required_columns - set(data.columns)
        print(f"File {file_name} is missing required columns: {missing_columns}. Skipping...")
        return None

    # Remove NaN values
    data = data.dropna().reset_index(drop=True)

    # Get stress change indices
    stress_change_indices = data.ne(data.shift()).apply(lambda x: x.index[x].tolist())["Stress"]
    stress_change_indices.append(len(data))

    # Create driving mode ranges
    labels = ["Rest1", "City1", "Highway1", "City2", "Highway2", "City3", "Rest2"]
    ranges = pd.cut(data.index, bins=stress_change_indices, labels=labels, right=False)

    # Filter data based on time range
    start_minute = int(SAMPLING_FREQUENCY * time_range[0] * 60)
    end_minute = int(SAMPLING_FREQUENCY * time_range[1] * 60)
    filtered_data = data.groupby(ranges).filter(lambda x: filter_minutes(x, start_minute, end_minute))

    # Normalize data
    columns_to_normalize = ["EMG", "RESP", "HR", "footGSR", "handGSR"]
    for col in columns_to_normalize:
        if col not in filtered_data.columns:
            print(f"Column '{col}' not found in {file_name}. Skipping normalization.")
            continue
        if col in ["footGSR", "handGSR"]:
            filtered_data[col] = (filtered_data[col] - filtered_data[col].min()) / (
                filtered_data[col].max() - filtered_data[col].min()
            )
        else:
            rest_mean = filtered_data.loc[ranges == "Rest1", col].mean()
            filtered_data[col] -= rest_mean

    # Add time column
    sampling_interval_ms = int(1 / SAMPLING_FREQUENCY * 1000)  # Convert sampling interval to milliseconds
    filtered_data["time"] = pd.date_range(
        start=0,
        periods=len(filtered_data),
        freq=f"{sampling_interval_ms}ms"
    )
    filtered_data.set_index("time", inplace=True)

    # Downsample and aggregate
    aggregated_data = filtered_data.groupby(ranges).resample(downsample_frequency).agg(functions_to_apply)

    # Flatten multi-level columns
    aggregated_data.columns = ["_".join(col).strip() for col in aggregated_data.columns.values]

    # Save processed data
    output_path = output_dir / f"{file_name}.csv"
    aggregated_data.reset_index(inplace=True)
    aggregated_data.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
    return aggregated_data

def main(input_path=INPUT_PATH, output_path=OUTPUT_PATH, downsample_frequency="10S", time_range=(3, 9)):
    """
    Main function to process all files in the input directory.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_path}")

    all_drives = []
    for file in Path(input_path).glob("*.csv"):
        if file.stem == "all_drives":
            continue  # Skip combined file if it exists
        processed_data = process_single_file(file, output_path, downsample_frequency, time_range)
        if processed_data is not None:
            processed_data["Drive"] = file.stem
            all_drives.append(processed_data)

    # Combine all driver data into one file
    if all_drives:
        combined_data = pd.concat(all_drives, ignore_index=True)
        combined_output_path = output_path / "all_drives.csv"
        combined_data.to_csv(combined_output_path, index=False)
        print(f"Combined data saved to: {combined_output_path}")

if __name__ == "__main__":
    main()