import pandas as pd
import numpy as np
import os

INPUT_CSV = "flex_sensor_data.csv"
OUTPUT_DIR = "augmented_data"
NUM_AUGMENTED = 10
NOISE_STD = 0.02
OFFSET_LIMIT = 0.05

def get_decimal_places(df: pd.DataFrame) -> int:
    max_decimals = 0
    for col in df.columns:
        for val in df[col].astype(str):
            if '.' in val:
                decimals = len(val.split('.')[1])
                if decimals > max_decimals:
                    max_decimals = decimals
    return max_decimals

def generate_synthetic_data(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    augmented_df = pd.DataFrame()
    num_rows = len(df)

    warp_factor = np.random.uniform(0.9, 1.1)
    new_indices = np.linspace(0, num_rows - 1, int(num_rows * warp_factor))

    for col in df.columns:
        warped_signal = np.interp(
            np.linspace(0, num_rows - 1, num_rows),
            new_indices,
            np.interp(new_indices, np.arange(num_rows), df[col])
        )

        noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=num_rows)
        shift = np.random.uniform(-OFFSET_LIMIT, OFFSET_LIMIT)

        synthetic_signal = warped_signal + noise + shift
        augmented_df[col] = np.round(synthetic_signal, decimals=precision)

    return augmented_df

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: '{INPUT_CSV}' not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    precision = get_decimal_places(df)
    print(f"Loaded '{INPUT_CSV}'. Detected decimal precision: {precision} decimal places.")

    for i in range(1, NUM_AUGMENTED + 1):
        synthetic_df = generate_synthetic_data(df, precision)
        output_filename = os.path.join(OUTPUT_DIR, f"augmented_flex_data_{i:02d}.csv")
        synthetic_df.to_csv(output_filename, index=False)
        print(f"Generated {output_filename}")

    print(f"\n created {NUM_AUGMENTED} files in '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
