
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

RUN_MODULE = "ai.behavior_detection.integration.run_video"

SAMPLE_FPS = 1


def start_detection(source):
    """Launch the existing HTTM AI V2 pipeline."""
    command = [
        PYTHON,
        "-m",
        RUN_MODULE,
        "--source",
        str(source),
        "--sample-fps",
        str(SAMPLE_FPS),
        "--pipeline",
        "legacy",
    ]

    try:
        subprocess.Popen(
            command,
            cwd=str(ROOT),
        )
    except Exception as exc:
        messagebox.showerror(
            "Loi khoi dong",
            str(exc),
        )


def choose_video():
    video_path = filedialog.askopenfilename(
        title="Chon video de phan tich",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*"),
        ],
    )

    if video_path:
        start_detection(video_path)


def open_webcam():
    start_detection(0)


def main():
    root = tk.Tk()

    root.title("HTTM - AI V2")
    root.geometry("460x340")
    root.resizable(False, False)

    tk.Label(
        root,
        text="HTTM - AI BEHAVIOR DETECTION",
        font=("Arial", 16, "bold"),
    ).pack(pady=(30, 8))

    tk.Label(
        root,
        text="Round 3 | Sleeping | Using Phone | AI target: 1 FPS",
        font=("Arial", 9),
    ).pack(pady=(0, 25))

    tk.Button(
        root,
        text="1. Tai video tu may tinh",
        font=("Arial", 12),
        width=28,
        height=2,
        command=choose_video,
    ).pack(pady=7)

    tk.Button(
        root,
        text="2. Mo webcam laptop",
        font=("Arial", 12),
        width=28,
        height=2,
        command=open_webcam,
    ).pack(pady=7)

    tk.Button(
        root,
        text="3. Thoat",
        font=("Arial", 11),
        width=28,
        command=root.destroy,
    ).pack(pady=15)

    root.mainloop()


if __name__ == "__main__":
    main()
