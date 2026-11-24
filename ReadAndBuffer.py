import serial
import serial.tools.list_ports
import time
import csv

# Configuration (Linux default fallback, will be auto-detected if possible)
SERIAL_PORT      = "/dev/ttyACM0"  # Change to /dev/ttyACM0 if your board uses that
BAUD_RATE        = 9600         # Match your Arduino/ESP32 baud rate
DURATION_SECONDS = 15.0             # How long to record
OUTPUT_CSV       = "flex_sensor_data.csv"

def find_serial_port() -> str:
    """Automatically search for Arduino / ESP32 serial ports on Linux."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # Check for typical Linux USB/ACM serial device patterns or descriptions
        if "USB" in p.device or "ACM" in p.device or "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return SERIAL_PORT

def main():
    port = find_serial_port()
    print(f"Connecting to {port} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)  # Allow time for microcontroller reset
    except Exception as e:
        print(f"Error opening serial port: {e}")
        print("Tip: Make sure you have permission to access the port (e.g., sudo usermod -aG dialout $USER)")
        return

    print(f"\n>>> RECORDING STARTED ({DURATION_SECONDS} seconds) <<<")
    print("Perform your gestures now...")

    data_rows = []
    start_time = time.time()
    ser.reset_input_buffer()

    # Collect data for the specified duration
    while time.time() - start_time < DURATION_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                # Filter out status messages and keep only lines starting with "DATA:"
                if line.startswith("DATA:"):
                    clean_line = line.replace("DATA:", "")
                    vals = [float(v) for v in clean_line.split(",")]
                    if len(vals) == 5:  # Ensure we have all 5 sensors
                        data_rows.append(vals)
            except Exception:
                # Skip malformed lines if a partial serial read happens
                pass

    ser.close()
    print(f"\n>>> RECORDING FINISHED. Captured {len(data_rows)} valid frames. <<<\n")

    # Save to CSV (containing ONLY sensor values and clean headers)
    print(f"Saving data to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Write clean column headers for the 5 sensors
        writer.writerow(["Sensor_0", "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"])
        # Write the pure numerical rows
        writer.writerows(data_rows)

    print(f"Successfully saved to {OUTPUT_CSV}!")

if __name__ == "__main__":
    main()
