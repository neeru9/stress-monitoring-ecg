import wfdb
import pandas as pd
import numpy as np

# List of driver records (base names without extensions)
driver_records = [
   'drive01', 'drive02', 'drive03', 'drive04', 'drive05',
    'drive06', 'drive07', 'drive08', 'drive09', 'drive10',
    'drive11', 'drive12', 'drive13', 'drive14', 'drive15',
    'drive16','drive17a','drive17b',
]

# Loop through each driver record
for record_name in driver_records:
    try:
        # Read the record (both .dat and .hea files)
        record = wfdb.rdrecord(record_name)

        # Extract signal data
        signals = record.p_signal

        # Get signal names from the record
        signal_names = record.sig_name

        # Create a DataFrame from the signal data
        df = pd.DataFrame(signals, columns=signal_names)

        # Calculate time values based on sampling frequency
        sampling_frequency = record.fs  # Sampling frequency in Hz
        num_samples = signals.shape[0]  # Number of samples
        time_values = np.arange(num_samples) / sampling_frequency  # Time in seconds

        # Add the time column to the DataFrame
        df.insert(0, 'Time', time_values)

        # Save the DataFrame to a CSV file
        csv_file_name = f"{record_name}.csv"
        df.to_csv(csv_file_name, index=False)

        print(f"Data for {record_name} saved to {csv_file_name}")

    except FileNotFoundError as e:
        print(f"Error: Could not find files for {record_name}. Skipping...")
        continue