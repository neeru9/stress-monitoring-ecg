from pathlib import Path
import pandas as pd

# Define paths
CSV_PATH = r"C:/Users/WILLIAM7771/Desktop/procc/predicting-driver-stress-using-deep-learning/data"
MARKER_DATA_CSV = r"C:/Users/WILLIAM7771/Desktop/procc/predicting-driver-stress-using-deep-learning/Marker Data/marker_info.csv"
OUTPUT_DIR = Path(CSV_PATH).joinpath("preprocessed_data")

def load_marker_data(marker_file_path):
    """
    Loads the marker data from the specified file.
    """
    if not marker_file_path.exists():
        raise FileNotFoundError(f"Marker file not found at: {marker_file_path}")
    print(f"Loading marker data from: {marker_file_path}")
    return pd.read_csv(marker_file_path)

def calculate_starting_times(marker_data):
    """
    Calculates starting times for each driving period.
    """
    print("Calculating starting times...")
    formatted_data = marker_data.copy()
    formatted_data['Rest1'] = 0
    numeric_cols = marker_data.iloc[:, 1:8].columns
    for index, column in enumerate(numeric_cols):
        if index < len(numeric_cols) - 1:
            next_column = numeric_cols[index + 1]
            formatted_data[next_column] = formatted_data[column] + marker_data[column]
    print("Starting times calculated successfully.")
    return formatted_data

def get_starting_indices(processed_markers, drive):
    """
    Converts starting times (in minutes) to indices based on a sampling rate of 15.5 Hz.
    """
    signals = processed_markers[processed_markers["Driver"] == drive].iloc[:, 1:8]
    indices = (signals * 15.5 * 60).astype(int).values[0]
    print(f"Starting indices for {drive}: {indices}")
    return indices

def label_data(row_index, starting_indices):
    """
    Labels the data based on the row index and starting indices.
    """
    relaxed = (row_index >= starting_indices[0] and row_index < starting_indices[1]) \
        or (row_index > starting_indices[6])
    medium = (row_index >= starting_indices[2] and row_index < starting_indices[3]) \
        or (row_index >= starting_indices[4] and row_index < starting_indices[5])
    stressed = (row_index >= starting_indices[1] and row_index < starting_indices[2]) \
        or (row_index >= starting_indices[3] and row_index < starting_indices[4]) \
        or (row_index >= starting_indices[5] and row_index < starting_indices[6])
    return 1.0 if relaxed else 3.0 if medium else 5.0

def process_driver_data(data, starting_indices, driver_name):
    """
    Processes raw data for a single driver by labeling it and saving it to a CSV file.
    """
    print(f"Processing data for {driver_name}...")
    data['Stress'] = data.apply(
        lambda row: label_data(row.name, starting_indices), axis=1
    )
    output_path = OUTPUT_DIR / f"{driver_name}.csv"
    print(f"Saving processed data to: {output_path}")
    data.to_csv(output_path, index=False)
    return data

def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")

    # Load marker data
    marker_file_path = Path(MARKER_DATA_CSV)
    marker_data = load_marker_data(marker_file_path)

    # Calculate starting times
    processed_markers = calculate_starting_times(marker_data)

    # Load raw CSV files
    csv_files = list(Path(CSV_PATH).glob("*.csv"))
    if not csv_files:
        print("No raw CSV files found in the input directory.")
        return

    csv_files_names = [Path(f).stem.lower() for f in csv_files]
    print(f"Found {len(csv_files_names)} raw CSV files: {csv_files_names}")

    # Process each driver's data
    all_drives = []
    for drive in processed_markers["Driver"]:
        if drive.lower() in csv_files_names:
            idx = csv_files_names.index(drive.lower())
            csv_file = csv_files[idx]
            print(f"\nProcessing driver: {drive}")
            data = pd.read_csv(csv_file)
            data = data.drop(['marker-mV'], axis=1, errors='ignore')

            # Get starting indices
            starting_indices = get_starting_indices(processed_markers, drive)

            # Process and save the data
            processed_data = process_driver_data(data, starting_indices, drive)
            processed_data['Drive'] = drive
            all_drives.append(processed_data)
        else:
            print(f"No matching CSV file found for driver: {drive}")

    # Combine all driver data into one file
    if all_drives:
        all_drives_data = pd.concat(all_drives, ignore_index=True)
        combined_output_path = OUTPUT_DIR / "all_drives.csv"
        print(f"Saving combined data to: {combined_output_path}")
        all_drives_data.to_csv(combined_output_path, index=False)
    else:
        print("No data to combine. Skipping creation of all_drives.csv.")

if __name__ == "__main__":
    main()