"""Estruturas de dados compartilhadas por todos os módulos.

Convenções:
- Coordenadas de imagem em pixels (x para a direita, y para baixo).
- Keypoints no formato COCO-17 (o mesmo do YOLO-pose).
- Slots de jogador relativos à câmera: "far_left", "far_right", "near_left", "near_right".
  "far" = lado da quadra do outro lado da rede (longe da câmera); "near" = lado perto da câmera.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Índices COCO-17
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16
WRISTS = (KP_LEFT_WRIST, KP_RIGHT_WRIST)
ANKLES = (KP_LEFT_ANKLE, KP_RIGHT_ANKLE)

SLOTS = ("far_left", "far_right", "near_left", "near_right")
TEAMS = ("far", "near")


def team_of(slot: str | None) -> str | None:
    """Dupla de um slot ("far_left" -> "far")."""
    if slot is None:
        return None
    return slot.split("_", 1)[0]


@dataclass
class PlayerObs:
    """Uma pessoa detectada em um frame."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    keypoints: np.ndarray  # (17, 2) px
    kp_conf: np.ndarray  # (17,)
    det_conf: float = 1.0
    # Preenchidos pela etapa de identidade (identity.py)
    slot: str | None = None
    court_xy: tuple[float, float] | None = None

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)

    @property
    def bbox_center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)

    def foot_point(self, min_conf: float = 0.3) -> np.ndarray:
        """Ponto de contato com o chão: média dos tornozelos confiáveis ou base da bbox."""
        good = [k for k in ANKLES if self.kp_conf[k] >= min_conf]
        if good:
            return self.keypoints[good].mean(axis=0)
        x1, _, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y2], dtype=float)

    def wrists(self, min_conf: float = 0.3) -> np.ndarray:
        """Punhos confiáveis, shape (n, 2) com n em {0, 1, 2}."""
        good = [k for k in WRISTS if self.kp_conf[k] >= min_conf]
        return self.keypoints[good] if good else np.empty((0, 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": [float(v) for v in self.bbox],
            "keypoints": np.asarray(self.keypoints, dtype=float).round(2).tolist(),
            "kp_conf": np.asarray(self.kp_conf, dtype=float).round(3).tolist(),
            "det_conf": round(float(self.det_conf), 3),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlayerObs":
        return cls(
            track_id=int(d["track_id"]),
            bbox=tuple(d["bbox"]),
            keypoints=np.asarray(d["keypoints"], dtype=float),
            kp_conf=np.asarray(d["kp_conf"], dtype=float),
            det_conf=float(d.get("det_conf", 1.0)),
        )


@dataclass
class BallObs:
    xy: tuple[float, float]
    conf: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"xy": [round(float(v), 2) for v in self.xy], "conf": round(float(self.conf), 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "BallObs | None":
        if not d:
            return None
        return cls(xy=tuple(d["xy"]), conf=float(d.get("conf", 1.0)))


@dataclass
class FrameObs:
    frame_idx: int
    players: list[PlayerObs] = field(default_factory=list)
    ball: BallObs | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "players": [p.to_dict() for p in self.players],
            "ball": self.ball.to_dict() if self.ball else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FrameObs":
        return cls(
            frame_idx=int(d["frame_idx"]),
            players=[PlayerObs.from_dict(p) for p in d["players"]],
            ball=BallObs.from_dict(d.get("ball")),
        )


@dataclass
class Hit:
    """Um golpe anotado (ground truth de tempo; jogador opcional)."""

    hit_id: str
    video_id: str
    frame: int
    gt_slot: str | None = None  # slot relativo à câmera, já normalizado pelo adapter
    rally_id: str | None = None

    @property
    def gt_team(self) -> str | None:
        return team_of(self.gt_slot)


@dataclass
class HitWindow:
    """Observações em torno de um golpe (o que fica salvo no cache)."""

    hit: Hit
    frames: list[FrameObs]

    def to_dict(self) -> dict[str, Any]:
        h = self.hit
        return {
            "hit": {
                "hit_id": h.hit_id,
                "video_id": h.video_id,
                "frame": h.frame,
                "gt_slot": h.gt_slot,
                "rally_id": h.rally_id,
            },
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HitWindow":
        return cls(hit=Hit(**d["hit"]), frames=[FrameObs.from_dict(f) for f in d["frames"]])
