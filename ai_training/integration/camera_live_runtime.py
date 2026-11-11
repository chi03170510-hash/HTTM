"""Camera Live lifecycle only; no inference, tracking or behavior decisions."""

import os
import queue
import subprocess
import sys
import threading
import time

from camera_coordinator import Worker


LIMIT_ENV = "HTTM_CAMERA_MAX_OBSERVATIONS"
RESULT_QUEUE_SIZE = 3  # Latest snapshots, with room for error + finished controls.
SHUTDOWN_TIMEOUT = 20.


def observation_limit_from_environment(default=None):
    value = os.environ.get(LIMIT_ENV)
    if value is None:
        return default
    if not value or not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"Invalid {LIMIT_ENV}={value!r}: use a positive integer; unset for continuous Camera Live")
    return int(value)


def publish_latest(results, item):
    """Never block inference on the UI. Only the display mailbox drops backlog.

    The producer publishes data first, then at most two error controls and one
    finished control. A mailbox of at least three preserves all terminal controls.
    Diagnostic writers still receive each completed AI observation directly.
    """
    while True:
        try:
            results.put_nowait(item)
            return
        except queue.Full:
            try:
                results.get_nowait()
            except queue.Empty:
                pass


class WorkerStopped(Exception):
    """Requested cancellation, not a model or camera failure."""


class _StopAwareResponses:
    def __init__(self, original, stop_event):
        self.original, self.stop_event = original, stop_event

    def put(self, item):
        return self.original.put(item)

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise WorkerStopped()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty()
            try:
                return self.original.get(timeout=min(.1, remaining))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise


class LiveWorker(Worker):
    """Reuse the existing worker protocol, with a cancellable response wait.

    The response proxy forwards to the SAME queue, including reader-thread puts;
    switching it cannot lose an already queued reply. Workers remain sequential.
    """
    stop_event = None

    def __init__(self, python_path, script_path):
        # A valid worker has one outstanding request. Reserve two reply slots
        # and one EOF slot; unexpected stdout must not accumulate indefinitely.
        self.responses = self._response_queue = queue.Queue(maxsize=3)
        self.process = self.reader = None
        self._ipc_failure = None
        self._closing = False
        try:
            self.process = subprocess.Popen(
                [str(python_path), "-u", str(script_path)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=None, text=True, encoding="utf-8", bufsize=1,
            )
            self.reader = threading.Thread(target=self._read_output, daemon=True)
            self.reader.start()
        except BaseException:
            # __init__ has not returned: ai_loop cannot own/close this process.
            try:
                self.close()
            except Exception as exc:
                print(f"[LIVE WORKER] initialization cleanup failed: {exc}", file=sys.stderr, flush=True)
            raise

    def _read_output(self):
        # start() can return before this target runs; process_frame may already
        # have installed its wait proxy. The producer always uses the real queue.
        responses = getattr(self, "_response_queue", self.responses)
        try:
            for line in self.process.stdout:
                if responses.qsize() >= 2:
                    self._ipc_failure = "Worker response backlog exceeded protocol limit"
                    break
                responses.put_nowait(("line", line))
        except (OSError, ValueError) as exc:
            if not self._closing:
                self._ipc_failure = f"Worker output reader failed: {exc}"
        finally:
            responses.put_nowait(("eof", None))

    def process_frame(self, image_path, frame_id):
        if self.stop_event is not None and self.stop_event.is_set():
            raise WorkerStopped()
        if getattr(self, "_ipc_failure", None):
            raise RuntimeError(self._ipc_failure)
        original = self.responses
        self.responses = _StopAwareResponses(original, self.stop_event)
        try:
            result = super().process_frame(image_path, frame_id)
            if getattr(self, "_ipc_failure", None):
                raise RuntimeError(self._ipc_failure)
            return result
        finally:
            self.responses = original

    def close(self):
        """Stop a local subprocess even when native inference is still busy."""
        process = self.process
        self._closing = True
        if process is None:
            return
        try:
            if process.poll() is None:
                # ai_loop sets stop before closing. No subsequent request exists;
                # native inference need not complete after Q/error/session end.
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        finally:
            # Close pipes only after the process stops, to avoid a pipe read lock
            # preventing cancellation. Cleanup of the second worker is independent.
            if process.poll() is not None:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
                reader = self.reader
                if reader is not None and reader is not threading.current_thread() and reader.ident is not None:
                    reader.join(timeout=1)
