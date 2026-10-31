"""Leitura de janelas de frames em torno de cada golpe e cache das observações."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .types import HitWindow


def iter_window(video_path: str | Path, center: int, pad: int) -> Iterator[tuple[int, np.ndarray]]:
    """Gera (frame_idx, frame) de center-pad até center+pad (inclusive)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = max(0, center - pad), min(total - 1, center + pad)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for idx in range(start, end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
    finally:
        cap.release()


def video_fps(video_path: str | Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps)


# ---------- cache JSONL (uma janela por linha) ----------
def append_cache(path: str | Path, window: HitWindow) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(window.to_dict()) + "\n")


def read_cache(path: str | Path) -> list[HitWindow]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(HitWindow.from_dict(json.loads(line)))
    return out


def cached_hit_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {w.hit.hit_id for w in read_cache(p)}
