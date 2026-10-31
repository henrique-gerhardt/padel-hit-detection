"""Estimativa de pose + rastreamento de pessoas.

A interface `PoseTracker` isola o modelo. Hoje: YOLO-pose (Ultralytics, licença AGPL-3.0 —
OK para a disciplina). Para o produto, implementar a mesma interface com RTMPose/ViTPose
(MMPose, Apache-2.0) sem mexer no resto da pipeline.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import PlayerObs


class PoseTracker(Protocol):
    def reset(self) -> None:
        """Zera o estado do rastreador (chamar no início de cada janela/rally)."""

    def __call__(self, frame_bgr: np.ndarray) -> list[PlayerObs]:
        """Pessoas detectadas no frame, com keypoints COCO-17 e track_id."""


class UltralyticsPoseTracker:
    def __init__(self, weights: str = "yolo11m-pose.pt", tracker: str = "bytetrack.yaml",
                 conf: float = 0.25, imgsz: int = 1280, device: str | None = None):
        from ultralytics import YOLO  # import tardio: o resto do projeto roda sem ultralytics

        self.model = YOLO(weights)
        self.tracker = tracker
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._fresh = True

    def reset(self) -> None:
        self._fresh = True

    def __call__(self, frame_bgr: np.ndarray) -> list[PlayerObs]:
        # persist=False no primeiro frame recria o rastreador; depois persist=True mantém os IDs.
        res = self.model.track(frame_bgr, persist=not self._fresh, tracker=self.tracker,
                               conf=self.conf, imgsz=self.imgsz, classes=[0],
                               device=self.device, verbose=False)[0]
        self._fresh = False
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
               else -np.arange(1, len(boxes) + 1))  # sem ID: negativos (não suavizáveis)
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
               else np.ones(kxy.shape[:2]))
        return [PlayerObs(int(ids[i]), tuple(boxes[i].tolist()), kxy[i], kcf[i], float(confs[i]))
                for i in range(len(boxes))]
