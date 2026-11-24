import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
import numpy as np
import time
import os
import subprocess
import sys
import threading
from collections import deque
import matplotlib.pyplot as plt
from model import FlexGloveCNN


SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 9600
NUM_SENSORS   = 5
WINDOW_SIZE   = 30
HOLD_SECONDS  = 3
SAMPLE_HZ     = 50
MODEL_PATH    = "glove_model.pt"

# Thread-safe buffer for live plotting
LIVE_BUFFER_SIZE = 150
live_data_queue = deque(np.zeros((LIVE_BUFFER_SIZE, NUM_SENSORS)), maxlen=LIVE_BUFFER_SIZE)
serial_lock = threading.Lock()

def load_target_words() -> list[str]:
    """Must match the exact class order used during training."""
    return ["a", "b"]

def find_serial_port() -> str:
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "USB" in p.device or "ACM" in p.device or "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return SERIAL_PORT

def background_serial_reader(ser: serial.Serial):
    """Continuously reads the serial port in the background."""
    global live_data_queue
    while ser.is_open:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("DATA:"):
                line = line.replace("DATA:", "")
            vals = [float(v) for v in line.split(",")]
            if len(vals) == NUM_SENSORS:
                with serial_lock:
                    live_data_queue.append(np.array(vals, dtype=np.float32))
        except Exception:
            pass

def start_live_plot():
    """Initializes a non-blocking live Matplotlib window for sensor inputs."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = []
    colors = ['r', 'g', 'b', 'c', 'm']
    for i in range(NUM_SENSORS):
        line, = ax.plot([], [], label=f"Sensor {i}", color=colors[i], linewidth=1.5)
        lines.append(line)

    ax.set_xlim(0, LIVE_BUFFER_SIZE)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Live Flex Sensor Data Stream (INFERENCE MODE)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Sensor Value")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig, ax, lines

def update_live_plot(fig, ax, lines):
    """Refreshes the live graph window."""
    with serial_lock:
        data = np.array(list(live_data_queue))
    if len(data) > 0:
        for i in range(NUM_SENSORS):
            lines[i].set_ydata(data[:, i])
            lines[i].set_xdata(np.arange(len(data)))
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw()
        fig.canvas.flush_events()

def collect_gesture(ser: serial.Serial, duration_s: float = HOLD_SECONDS, fig=None, ax=None, lines=None) -> np.ndarray:
    """Collects gesture window while keeping the live graph updating."""
    ser.reset_input_buffer()
    frames = []
    deadline = time.time() + duration_s
    while time.time() < deadline:
        with serial_lock:
            if len(live_data_queue) > 0:
                frames.append(live_data_queue[-1])
        if fig and ax and lines:
            update_live_plot(fig, ax, lines)
        time.sleep(1.0 / SAMPLE_HZ)

    if len(frames) >= WINDOW_SIZE:
        start = (len(frames) - WINDOW_SIZE) // 2
        frames = frames[start:start + WINDOW_SIZE]
    else:
        while len(frames) < WINDOW_SIZE:
            frames.append(frames[-1] if frames else np.zeros(NUM_SENSORS))
    return np.array(frames, dtype=np.float32)

def normalize(window: np.ndarray) -> np.ndarray:
    mn = window.min(axis=0, keepdims=True)
    mx = window.max(axis=0, keepdims=True)
    rng = np.where((mx - mn) < 1e-6, 1.0, mx - mn)
    return (window - mn) / rng

def predict(model: nn.Module, window: np.ndarray, device: torch.device) -> tuple[int, float]:
    """Runs the exact forward pass minus backpropagation."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(normalize(window)).unsqueeze(0).to(device)
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
        return pred_idx.item(), confidence.item()

def print_banner(text: str, char: str = "─") -> None:
    try:
        w = min(60, os.get_terminal_size().columns)
    except OSError:
        w = 60
    print(char * w)
    print(text.center(w))
    print(char * w)

def main():
    print_banner("FLEX GLOVE INFERENCE (TEST MODE)", "═")

    word_list = load_target_words()
    num_classes = len(word_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and weights
    model = FlexGloveCNN(NUM_SENSORS, WINDOW_SIZE, num_classes).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded trained model from {MODEL_PATH}")
    else:
        print(f"Error: No model found at '{MODEL_PATH}'. Train the model first!")
        return

    port = find_serial_port()
    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("Connected. Opening live graph window...\n")

    reader_thread = threading.Thread(target=background_serial_reader, args=(ser,), daemon=True)
    reader_thread.start()

    fig, ax, lines = start_live_plot()

    try:
        while True:
            print_banner("Ready to predict! Sign 'A' or 'B'...")
            print(f"You have {HOLD_SECONDS} seconds. Starting in 2s...")

            countdown_end = time.time() + 2.0
            while time.time() < countdown_end:
                update_live_plot(fig, ax, lines)
                time.sleep(0.02)

            print(">>> RECORDING GESTURE <<<")
            window = collect_gesture(ser, HOLD_SECONDS, fig, ax, lines)

            # Pure inference forward pass (no gradients, no backprop)
            pred_idx, confidence = predict(model, window, device)
            predicted_word = word_list[pred_idx]
            subprocess.run([sys.executable, "speak.py", predicted_word.upper()])
            print(f"\n  Prediction : {predicted_word.upper()}  ({confidence * 100:.1f}% confidence)\n")

            print("Press Enter for next test, or Ctrl+C to quit...")
            input()

    except KeyboardInterrupt:
        print("\nInference ended.")
        ser.close()
        plt.close('all')

if __name__ == "__main__":
    main()
