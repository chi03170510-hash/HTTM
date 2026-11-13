"""Compatibility shim delegating the legacy worker entrypoint to SmartClassroomStream."""

from smart_classroom_stream_worker import SmartClassroomStream, main

HybridV7Stream = SmartClassroomStream

if __name__ == "__main__":
    main()
