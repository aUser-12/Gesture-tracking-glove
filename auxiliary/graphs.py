import pandas as pd
import matplotlib.pyplot as plt

INPUT_CSV = str(input())

def main():
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: '{INPUT_CSV}' not found.")
        return

    plt.figure(figsize=(12, 6))

    for col in df.columns:
        plt.plot(df.index, df[col], label=col, linewidth=1.5)

    plt.title("Flex Sensor Data Streams (5 Sensors)")
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Sensor Value", fontsize=12)

    plt.legend(title="Sensors", loc="upper right")
    plt.tight_layout()

    print("Displaying graph")
    plt.show()

if __name__ == "__main__":
    main()
