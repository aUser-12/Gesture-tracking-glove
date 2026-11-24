import serial
import serial.tools.list_ports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import subprocess
import time
import os
import sys
import threading
from collections import deque
import matplotlib.pyplot as plt
from model import FlexGloveCNN

# Configuration
SERIAL_PORT   = "/dev/ttyUSB0"   # Update to your port if needed
BAUD_RATE     = 9600        # Recommended stable baud rate (adjust if your ESP32 is different)
NUM_SENSORS   = 5
WINDOW_SIZE   = 30
HOLD_SECONDS  = 3
SAMPLE_HZ     = 50
SAVE_PATH     = "glove_model.pt"
LR            = 1e-3

# Thread-safe buffer for live plotting (holds the last 150 frames / ~3 seconds of data)
LIVE_BUFFER_SIZE = 150
live_data_queue = deque(np.zeros((LIVE_BUFFER_SIZE, NUM_SENSORS)), maxlen=LIVE_BUFFER_SIZE)
serial_lock = threading.Lock()

def load_target_words() -> list[str]:
    """Explicitly returns the two words you want to train on."""
    return ["a", "b"]

def find_serial_port() -> str:
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "USB" in p.device or "ACM" in p.device or "Arduino" in p.description or "CH340" in p.description:
            return p.device
    return SERIAL_PORT

def background_serial_reader(ser: serial.Serial):
    """Continuously reads the serial port in the background to feed the live graph."""
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
    ax.set_ylim(-20, 20)  # Adjust based on your sensor normalization or raw limits
    ax.set_title("Live Flex Sensor Data Stream", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("Sensor Value")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig, ax, lines

def update_live_plot(fig, ax, lines):
    """Refreshes the live graph window with the latest buffer contents."""
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

def train_step(
    model: FlexGloveCNN,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    window: np.ndarray,
    label_idx: int,
    device: torch.device,
) -> tuple[float, int]:
    model.train()
    x = torch.tensor(normalize(window)).unsqueeze(0).to(device)
    y = torch.tensor([label_idx], dtype=torch.long).to(device)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    pred = logits.argmax(dim=1).item()
    return loss.item(), pred

def print_banner(text: str, char: str = "─") -> None:
    try:
        w = min(60, os.get_terminal_size().columns)
    except OSError:
        w = 60
    print(char * w)
    print(text.center(w))
    print(char * w)

def main():
    print_banner("FLEX GLOVE TRAINER (A vs B)", "═")

    word_list = load_target_words()
    num_classes = len(word_list)
    print(f"Training classes: {word_list}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FlexGloveCNN(NUM_SENSORS, WINDOW_SIZE, num_classes).to(device)

    if os.path.exists(SAVE_PATH):
        try:
            model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
            print(f"Loaded existing model from {SAVE_PATH}")
        except Exception:
            print("Existing model shape mismatched. Starting fresh model...")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    port = find_serial_port()
    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("Connected. Opening live graph window...\n")

    # Start background serial reader thread
    reader_thread = threading.Thread(target=background_serial_reader, args=(ser,), daemon=True)
    reader_thread.start()

    # Initialize live graph window
    fig, ax, lines = start_live_plot()

    session_correct = 0
    session_total = 0

    try:
        while True:
            label_idx = random.randint(0, num_classes - 1)
            target_word = word_list[label_idx]

            print_banner(f'Sign the gesture for: "{target_word.upper()}"')
            if os.path.exists("speak.py"):
                try:
                    subprocess.run([sys.executable, "speak.py", target_word.upper()])
                except Exception:
                    pass

            print(f"You have {HOLD_SECONDS} seconds. Starting in 2s...")
            # Keep updating the graph during the countdown
            countdown_end = time.time() + 2.0
            while time.time() < countdown_end:
                update_live_plot(fig, ax, lines)
                time.sleep(0.02)

            print(">>> RECORDING <<<")
            window = collect_gesture(ser, HOLD_SECONDS, fig, ax, lines)

            print("Processing...")
            loss, pred_idx = train_step(model, optimizer, criterion, window, label_idx, device)
            pred_word = word_list[pred_idx]
            correct = pred_idx == label_idx

            session_total += 1
            session_correct += int(correct)
            accuracy = session_correct / session_total * 100

            print(f"\n  Target  : {target_word.upper()}")
            print(f"  Model   : {pred_word.upper()}  {'✓ CORRECT' if correct else '✗ wrong'}")
            print(f"  Loss    : {loss:.4f}")
            print(f"  Accuracy: {accuracy:.1f}%  ({session_correct}/{session_total})")

            torch.save(model.state_dict(), SAVE_PATH)

            print("\nPress Enter in terminal for next gesture, or Ctrl+C to quit...")
            # Continue refreshing graph while waiting for user input
            while True:
                update_live_plot(fig, ax, lines)
                if sys.stdin in select_stdin() if 'select_stdin' in globals() else True:
                    # Non-blocking check or simple break on input
                    break
                time.sleep(0.05)
                # To keep it simple with standard input() block, we refresh right before input:
                break
            input()

    except KeyboardInterrupt:
        print("\nSession ended.")
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Model saved to {SAVE_PATH}")
        ser.close()
        plt.close('all')

if __name__ == "__main__":
    main()
