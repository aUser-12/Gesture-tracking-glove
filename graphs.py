import pandas as pd
import matplotlib.pyplot as plt

# Configuration
INPUT_CSV = str(input())

def main():
    try:
        # Load the CSV file containing the sensor data
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: '{INPUT_CSV}' not found. Make sure you run the data collection script first!")
        return

    # Set up the plot window
    plt.figure(figsize=(12, 6))

    # Plot each sensor column on the same graph
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=1.5)

    # Graph styling and labels
    plt.title("Flex Sensor Data Streams (5 Sensors)", fontsize=14, fontweight="bold")
    plt.xlabel("Sample Frame Index", fontsize=12)
    plt.ylabel("Sensor Value", fontsize=12)

    # Add legends, grid lines, and tidy up layout
    plt.legend(title="Sensors", loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Display the interactive window
    print("Displaying graph...")
    plt.show()

if __name__ == "__main__":
    main()
