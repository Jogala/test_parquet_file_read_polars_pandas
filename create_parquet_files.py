# %%
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

# Configuration
NUM_FILES = 1
ENTRIES_PER_FILE = 20
LIST_LENGTH = 1_500_000
OUTPUT_DIR = Path("./synthetic_parquet_files")
cache = True
# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Function to generate a single parquet file
def generate_parquet_file(file_idx: int, cache: bool) -> tuple[Path, Path]:
    file_path_pandas = OUTPUT_DIR / "pandas" / f"data_{file_idx:03d}.parquet"
    file_path_polars = OUTPUT_DIR / "polars" / f"data_{file_idx:03d}.parquet"

    if file_path_pandas.exists() and file_path_polars.exists() and cache:
        return file_path_pandas, file_path_polars

    # Create start date (each file will have consecutive dates)
    start_date = datetime(2023, 1, 1) + timedelta(days=file_idx * ENTRIES_PER_FILE)

    # Generate dates
    start_unix_time = int(start_date.timestamp())
    dates = [start_unix_time + (i * 86400) for i in range(ENTRIES_PER_FILE)]

    # Generate values - each is a list of 1.5M floats
    values = []
    for i in range(ENTRIES_PER_FILE):
        # Use a deterministic seed based on file and entry index for reproducibility
        seed = file_idx * 1000 + i
        np.random.seed(seed)

        # Generate array with some patterns to make it more realistic:
        # - Base signal: sine wave with random frequency
        # - Add some noise
        # - Add occasional spikes

        x = np.linspace(0, 10 * np.pi, LIST_LENGTH)
        freq = 0.5 + np.random.random() * 2  # Random frequency between 0.5 and 2.5
        base_signal = np.sin(freq * x) * 10
        noise = np.random.normal(0, 1, LIST_LENGTH)

        # Add occasional spikes (about 0.1% of points)
        spike_mask = np.random.random(LIST_LENGTH) < 0.001
        spikes = np.zeros(LIST_LENGTH)
        spikes[spike_mask] = (np.random.random(sum(spike_mask)) * 40) - 20  # Spikes between -20 and 20

        # Combine components
        signal = base_signal + noise + spikes
        values.append(signal.tolist())

    # Create pandas DataFrame
    df = pd.DataFrame({"date": dates, "value": values, "value2": (10 * np.asarray(values)).tolist()})

    # Save to parquet
    file_path_pandas.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path_pandas, index=False)

    file_path_polars.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(df).write_parquet(file_path_polars)

    return file_path_pandas, file_path_polars


# Generate all files with progress bar
print(f"Generating {NUM_FILES} parquet files with {ENTRIES_PER_FILE} entries each...")
print(f"Each entry contains a list of {LIST_LENGTH:,} floats")
print(f"Output directory: {OUTPUT_DIR}")

# Calculate approximate size
single_list_size_mb = LIST_LENGTH * 4 / (1024 * 1024)  # 4 bytes per float
total_size_gb = NUM_FILES * ENTRIES_PER_FILE * single_list_size_mb / 1024
print(f"Estimated total size: {total_size_gb:.2f} GB (uncompressed)")

# Confirm before proceeding
confirmation = print(f"This will generate approximately {total_size_gb:.2f} GB of data.")

# Generate files with progress bar
file_paths = {"pandas": [], "polars": []}

for i in tqdm(range(NUM_FILES), desc="Generating files"):
    (file_path_pandas, file_path_polars) = generate_parquet_file(i, cache=cache)
    file_paths["pandas"].append(file_path_pandas)
    file_paths["polars"].append(file_path_polars)

for key, paths in file_paths.items():
    print("\n########################################################")
    print(f"{key}")
    print("########################################################")
    print(f"\nGenerated {len(paths)} parquet files in {OUTPUT_DIR}")

    # Verify one file to confirm structure
    print("\nVerifying structure of first file...")
    test_df = pd.read_parquet(paths[0])
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {test_df.columns.tolist()}")
    print(f"First date: {test_df['date'].iloc[0]}")
    print(f"Length of first value array: {len(test_df['value'].iloc[0])}")

    # Try to load with polars as well to ensure compatibility
    try:
        pl_df = pl.read_parquet(paths[0])
        print("\nSuccessfully loaded with Polars")
        print(f"Polars shape: {pl_df.shape}")
    except Exception as e:
        print(f"\nError loading with Polars: {e}")

    # Check original file
    meta = pq.read_metadata(paths[0])
    print(meta)
