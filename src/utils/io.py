from __future__ import annotations
import json, os, pathlib, random, time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, List

def ensure_dir(p: str) -> str:
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)
    return p

def jsonl_write(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def jsonl_read(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)

def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
