"""Small JSON logging helpers for training runs."""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def utc_timestamp() -> str:
    """Return a compact UTC timestamp for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def make_log_paths(run_name: str, log_dir: str = "outputs/logs") -> Dict[str, str]:
    """Create stable log and summary paths for a run."""
    os.makedirs(log_dir, exist_ok=True)
    stamp = utc_timestamp()
    return {
        "events": os.path.join(log_dir, f"{run_name}_{stamp}.jsonl"),
        "summary": os.path.join(log_dir, f"{run_name}_{stamp}_summary.json"),
    }


def append_jsonl(path: str, event: Dict[str, Any]) -> None:
    """Append one JSON event to a JSONL file."""
    record = {"timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"), **event}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with deterministic formatting."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def log_and_print(message: str, events_path: Optional[str] = None, **event: Any) -> None:
    """Print a message and optionally mirror a structured event to JSONL."""
    print(message)
    if events_path is not None:
        append_jsonl(events_path, {"message": message, **event})
