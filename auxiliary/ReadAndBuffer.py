import serial
import serial.tools.list_ports
import time
import csv

SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
DURATION_SECONDS = 15.0
OUTPUT_CSV = "flex_sensor_data.csv"

def find_serial_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "USB" in p.device or "ACM" in p.device or "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return SERIAL_PORT

def main():
    port = find_serial_port()
    print(f"Connecting to {port} at {BAUD_RATE} baud")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    print(f"\nRecording ({DURATION_SECONDS} seconds")
    print("Perform  gestures")

    data_rows = []
    start_time = time.time()
    ser.reset_input_buffer()

    while time.time() - start_time < DURATION_SECONDS:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()

                if line.startswith("DATA:"):
                    clean_line = line.replace("DATA:", "")
                    vals = [float(v) for v in clean_line.split(",")]
                    if len(vals) == 5:
                        data_rows.append(vals)
            except Exception:
                pass

    ser.close()
    print(f"\nRecorded. Captured {len(data_rows)} valid frames.\n")

    print(f"Saving data to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sensor_0", "Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"])
        writer.writerows(data_rows)

    print(f"saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
