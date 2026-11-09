"""Opt-in structured diagnostics, separate from detection/event JSONL stdout."""

import json
import sys


def emit_debug(event, **fields):
    print(json.dumps({"event": event, **fields}, ensure_ascii=False),
          file=sys.stderr, flush=True)
