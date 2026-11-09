
"""
HTTM - Classroom Behavior Monitoring
Video Source Selection

1. Video file from laptop (Windows file picker)
2. Laptop webcam
3. DroidCam

Runs the existing HTTM integration pipeline.
"""

import argparse
import subprocess
import sys
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import filedialog


# Project root: D:/Study/HTTM/HTTM
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLE_FPS = 1
MAX_DURATION = 15
PIPELINE = "legacy"

WEBCAM_ID = 0
DROIDCAM_ID = 1


def choose_video_file():
    """Open the Windows file picker to select a video."""

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        video_path = filedialog.askopenfilename(
            parent=root,
            title="HTTM - Chon video tu may tinh",
            initialdir=str(Path.home()),
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.wmv"
                ),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not video_path:
        return None

    path = Path(video_path)

    if not path.is_file():
        print("[ERROR] Video file does not exist.")
        return None

    print(f"\n[HTTM] Selected video: {path}")
    return str(path.resolve())


def show_menu():
    """Display video source selection menu."""

    print("\n" + "=" * 50)
    print("           HTTM - CHON NGUON VIDEO")
    print("=" * 50)

    print("1. Video co san tren laptop")
    print("2. Webcam laptop")
    print("3. DroidCam (camera dien thoai)")
    print("0. Thoat")

    print("=" * 50)


def select_source():
    """Let the user select a video source."""

    while True:
        show_menu()

        choice = input(
            "\nNhap lua chon (0-3): "
        ).strip()

        if choice == "0":
            return None, None

        if choice == "1":
            source = choose_video_file()

            if source is None:
                print(
                    "\n[HTTM] Chua chon video. "
                    "Quay lai menu."
                )
                continue

            return source, "video"

        if choice == "2":
            return str(WEBCAM_ID), "webcam"

        if choice == "3":
            return str(DROIDCAM_ID), "droidcam"

        print(
            "\n[ERROR] Lua chon khong hop le. "
            "Vui long chon lai."
        )


def create_output_paths(source_type):
    """Create unique output paths for each run."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"httm_{source_type}_{timestamp}"

    output_path = RESULTS_DIR / f"{filename}.mp4"
    events_path = RESULTS_DIR / f"{filename}.jsonl"

    return output_path, events_path


def run_pipeline(source, source_type, *, ch20=False):
    """Start the existing HTTM video pipeline."""

    output_path, events_path = create_output_paths(
        source_type
    )

    command = [
        sys.executable,
        "-m",
        "ai.behavior_detection.integration.run_video",
        "--source",
        str(source),
        "--source-type",
        source_type,
        "--sample-fps",
        str(SAMPLE_FPS),
        "--max-duration",
        str(MAX_DURATION),
        "--pipeline",
        PIPELINE,
        "--output",
        str(output_path),
        "--events",
        str(events_path),
    ]
    if ch20:
        command.extend(["--ch20", "--warmup"])

    print("\n" + "=" * 50)
    print("              HTTM STARTING")
    print("=" * 50)

    print(f"[HTTM] Source type: {source_type}")
    print(f"[HTTM] Source: {source}")
    print("[HTTM] Model: Round 3 - best.pt")
    print(f"[HTTM] Pipeline: {PIPELINE}")
    if ch20:
        print("[HTTM] CH-20: Student ID / Track ID / Behavior / Duration")
    print(f"[HTTM] AI target: {SAMPLE_FPS} FPS (video plays independently)")
    print(f"[HTTM] Max duration: {MAX_DURATION}s")

    print(f"\n[HTTM] Output: {output_path}")
    print(f"[HTTM] Events: {events_path}")

    print("\n[HTTM] Running...\n")

    try:
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            check=True,
        )

        print("\n" + "=" * 50)
        print("           HTTM COMPLETED")
        print("=" * 50)

        print(f"Video: {output_path}")
        print(f"Events: {events_path}")

    except subprocess.CalledProcessError as exc:
        print(
            f"\n[ERROR] Pipeline exited "
            f"with code {exc.returncode}"
        )

    except KeyboardInterrupt:
        print("\n[HTTM] Stopped by user.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="HTTM video/webcam/DroidCam source menu")
    parser.add_argument("--ch20", action="store_true", help="Enable per-student behavior/duration overlay")
    args = parser.parse_args(argv)
    print("\nWelcome to HTTM")

    while True:
        source, source_type = select_source()

        if source is None:
            print("\n[HTTM] Goodbye!")
            break

        run_pipeline(source, source_type, ch20=args.ch20)

        again = input(
            "\nBan co muon chay tiep? (y/n): "
        ).strip().lower()

        if again not in ("y", "yes"):
            print("\n[HTTM] Goodbye!")
            break


if __name__ == "__main__":
    main()
