"""Small helpers for the eval metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield one dict per non-blank line in a .jsonl file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)


__all__ = ["load_jsonl"]
