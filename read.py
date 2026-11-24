
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import serial


class FlexSensorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flex Sensor Monitor")
        self.root.geometry("400x300")

        self.serial_port = None
        self.is_running = False

        # Serial port input
        ttk.Label(root, text="Serial Port:").pack(pady=(10, 0))
        self.port_entry = ttk.Entry(root)
        self.port_entry.insert(0, "/dev/ttyACM0")
        self.port_entry.pack()

        # Baud rate input
        ttk.Label(root, text="Baud Rate:").pack(pady=(10, 0))
        self.baud_entry = ttk.Entry(root)
        self.baud_entry.insert(0, 9600)
        self.baud_entry.pack()

        # Five flex sensor labels
        self.sensor_labels = []
        for i in range(5):
            label = ttk.Label(root, text=f"Flex Sensor {i + 1}: --")
            label.pack(pady=2)
            self.sensor_labels.append(label)

        # Start button
        self.start_button = ttk.Button(
            root, text="Start", command=self.start_reading
        )
        self.start_button.pack(pady=10)

    def start_reading(self):
        if self.is_running:
            return

        port = self.port_entry.get().strip()

        try:
            baud_rate = int(self.baud_entry.get().strip())
            if baud_rate <= 0:
                raise ValueError("Baud rate must be positive.")

            self.serial_port = serial.Serial(
                port, baud_rate, timeout=1
            )

            self.is_running = True
            self.start_button.config(state="disabled")

            threading.Thread(
                target=self.read_serial,
                daemon=True
            ).start()

        except (ValueError, serial.SerialException) as e:
            messagebox.showerror("Connection Error", str(e))

    def read_serial(self):
        try:
            while self.is_running:
                line = self.serial_port.readline().decode(
                    "utf-8", errors="ignore"
                ).strip()

                if line:
                    print(line, flush=True)

                    # TODO: Parse sensor values here.
                    # TODO: Update the five sensor labels here.
        except (serial.SerialException, OSError) as e:
            print(f"Serial error: {e}")
        finally:
            self.is_running = False
            if self.serial_port and self.serial_port.is_open:
                self.serial_port.close()

            self.root.after(0, lambda: self.start_button.config(
                state="normal"
            ))


if __name__ == "__main__":
    root = tk.Tk()
    app = FlexSensorApp(root)
    root.mainloop()
