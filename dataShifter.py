import pandas as pd
import numpy as np
import os

# Configuration
INPUT_CSV       = "flex_sensor_data.csv"
OUTPUT_DIR      = "augmented_data"
NUM_AUGMENTED   = 10   # Number of new CSV files to generate
NOISE_STD       = 0.02 # Standard deviation of random Gaussian noise
OFFSET_LIMIT    = 0.05 # Max baseline shift (simulates slight sensor placement change)

def get_decimal_places(df: pd.DataFrame) -> int:
    """Determine the maximum decimal precision present in the original dataset."""
    max_decimals = 0
    for col in df.columns:
        for val in df[col].astype(str):
            if '.' in val:
                decimals = len(val.split('.')[1])
                if decimals > max_decimals:
                    max_decimals = decimals
    return max_decimals

def generate_synthetic_data(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    """Applies noise, baseline shift, and subtle time-warping to generate a plausible variation."""
    augmented_df = pd.DataFrame()
    num_rows = len(df)

    # 1. Subtle Time-Warping (simulates slight speed variation in gesture execution)
    # Creates an interpolated index to stretch/compress time slightly (between 90% and 110% speed)
    warp_factor = np.random.uniform(0.9, 1.1)
    new_indices = np.linspace(0, num_rows - 1, int(num_rows * warp_factor))

    for col in df.columns:
        # Interpolate across the warped time timeline
        warped_signal = np.interate if hasattr(np, 'interate') else np.interp(
            np.linspace(0, num_rows - 1, num_rows),
            new_indices,
            np.interp(new_indices, np.arange(num_rows), df[col])
        )

        # 2. Add Random Gaussian Noise (jitter)
        noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=num_rows)

        # 3. Add Random Baseline Offset/Shift (simulates minor glove fitting changes)
        shift = np.random.uniform(-OFFSET_LIMIT, OFFSET_LIMIT)

        # Combine transformations
        synthetic_signal = warped_signal + noise + shift

        # Preserve original precision by rounding
        augmented_df[col] = np.round(synthetic_signal, decimals=precision)

    return augmented_df

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: '{INPUT_CSV}' not found. Run the collection script first!")
        return

    # Create output directory for synthetic files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load original dataset
    df = pd.read_csv(INPUT_CSV)
    precision = get_decimal_places(df)
    print(f"Loaded '{INPUT_CSV}'. Detected decimal precision: {precision} decimal places.")

    # Generate synthetic CSV files
    for i in range(1, NUM_AUGMENTED + 1):
        synthetic_df = generate_synthetic_data(df, precision)
        output_filename = os.path.join(OUTPUT_DIR, f"augmented_flex_data_{i:02d}.csv")
        synthetic_df.to_csv(output_filename, index=False)
        print(f"Generated: {output_filename}")

    print(f"\nSuccessfully created {NUM_AUGMENTED} synthetic CSV files in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
