"""Detecção da bola.

Implementações:
- NullBallDetector: sem bola (força o fallback; útil para ablação).
- PrecomputedBallDetector: lê trajetórias já calculadas (CSV video_id,frame,x,y,conf),
  p.ex. geradas por um repositório TrackNet externo rodado à parte. É o caminho recomendado:
  rodar o TrackNet uma vez, salvar CSV e iterar rápido na atribuição.
- TrackNetBallDetector: esqueleto para integrar o TrackNet em processo (TODO).
"""
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import BallObs


class BallDetector(Protocol):
    def reset(self) -> None: ...

    def __call__(self, frame_bgr: np.ndarray, frame_idx: int, video_id: str) -> BallObs | None: ...


class NullBallDetector:
    def reset(self) -> None:
        pass

    def __call__(self, frame_bgr, frame_idx, video_id):
        return None


class PrecomputedBallDetector:
    def __init__(self, csv_path: str | Path, min_conf: float = 0.0):
        self.table: dict[str, dict[int, BallObs]] = defaultdict(dict)
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("x") in (None, "", "nan"):
                    continue
                conf = float(row.get("conf") or 1.0)
                if conf >= min_conf:
                    self.table[row["video_id"]][int(row["frame"])] = BallObs(
                        (float(row["x"]), float(row["y"])), conf)

    def reset(self) -> None:
        pass

    def __call__(self, frame_bgr, frame_idx, video_id):
        return self.table.get(video_id, {}).get(frame_idx)


class TrackNetBallDetector:
    """TODO: integrar um TrackNet pré-treinado (tênis) — o artigo usa TrackNet sem fine-tuning.

    Sugestão: clonar um repositório TrackNet (PyTorch) com pesos de tênis, e implementar
    `_infer(frames)` que recebe os 3 últimos frames (redimensionados para 640x360) e devolve
    o heatmap; o pico do heatmap acima de um limiar vira a posição da bola (reescalar para o
    tamanho original). Alternativa mais simples: rodar o repositório externo sobre o vídeo,
    exportar CSV e usar PrecomputedBallDetector.
    """

    def __init__(self, weights: str, n_frames: int = 3, threshold: float = 0.5):
        self.weights, self.n_frames, self.threshold = weights, n_frames, threshold
        self.buffer: deque[np.ndarray] = deque(maxlen=n_frames)

    def reset(self) -> None:
        self.buffer.clear()

    def __call__(self, frame_bgr, frame_idx, video_id):
        self.buffer.append(frame_bgr)
        if len(self.buffer) < self.n_frames:
            return None
        return self._infer(list(self.buffer))

    def _infer(self, frames: list[np.ndarray]) -> BallObs | None:
        raise NotImplementedError("Integrar modelo TrackNet aqui (ver docstring da classe).")
