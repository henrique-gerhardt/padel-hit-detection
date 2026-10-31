"""Modelo da quadra: homografia imagem -> plano da quadra (metros).

Sistema de coordenadas da quadra (vista de cima, a partir da câmera):
    x: 0 (parede lateral esquerda) .. 10 (parede lateral direita)
    y: 0 (fundo LONGE da câmera) .. 20 (fundo PERTO da câmera); rede em y = 10.
Linhas de saque a 6,95 m da rede -> y = 3,05 e y = 16,95; linha central de saque em x = 5.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

COURT_W = 10.0
COURT_L = 20.0
NET_Y = 10.0
SERVICE_DIST = 6.95

# Pontos de referência nomeados (metros). A calibração clica alguns deles na imagem.
REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "far_left_corner": (0.0, 0.0),
    "far_right_corner": (COURT_W, 0.0),
    "near_right_corner": (COURT_W, COURT_L),
    "near_left_corner": (0.0, COURT_L),
    "far_service_left": (0.0, NET_Y - SERVICE_DIST),
    "far_service_right": (COURT_W, NET_Y - SERVICE_DIST),
    "far_service_center": (COURT_W / 2, NET_Y - SERVICE_DIST),
    "near_service_left": (0.0, NET_Y + SERVICE_DIST),
    "near_service_right": (COURT_W, NET_Y + SERVICE_DIST),
    "near_service_center": (COURT_W / 2, NET_Y + SERVICE_DIST),
    "net_left": (0.0, NET_Y),
    "net_right": (COURT_W, NET_Y),
}


class CourtModel:
    def __init__(self, H_img2court: np.ndarray):
        self.H = np.asarray(H_img2court, dtype=float)
        self.H_inv = np.linalg.inv(self.H)

    # ---------- construção / persistência ----------
    @classmethod
    def from_correspondences(cls, img_pts: np.ndarray, court_pts: np.ndarray) -> "CourtModel":
        img_pts = np.asarray(img_pts, dtype=np.float32)
        court_pts = np.asarray(court_pts, dtype=np.float32)
        if len(img_pts) < 4:
            raise ValueError("São necessários pelo menos 4 pontos para a homografia.")
        method = cv2.RANSAC if len(img_pts) > 4 else 0
        H, _ = cv2.findHomography(img_pts, court_pts, method, 3.0)
        if H is None:
            raise ValueError("Homografia degenerada; confira os pontos clicados.")
        return cls(H)

    @classmethod
    def from_named_points(cls, named_img_pts: dict[str, list[float]]) -> "CourtModel":
        names = [n for n in named_img_pts if n in REFERENCE_POINTS]
        img = np.array([named_img_pts[n] for n in names])
        court = np.array([REFERENCE_POINTS[n] for n in names])
        return cls.from_correspondences(img, court)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        data = {"H_img2court": self.H.tolist(), **(extra or {})}
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CourtModel":
        data = json.loads(Path(path).read_text())
        if "H_img2court" in data:
            return cls(np.array(data["H_img2court"]))
        return cls.from_named_points(data["points"])

    # ---------- transformações ----------
    def to_court(self, pts_img: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_img, dtype=float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts_court: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_court, dtype=float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    # ---------- consultas ----------
    @staticmethod
    def in_court(xy: np.ndarray, margin: float = 0.5) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return -margin <= x <= COURT_W + margin and -margin <= y <= COURT_L + margin

    @staticmethod
    def side(xy: np.ndarray) -> str:
        return "far" if float(xy[1]) < NET_Y else "near"

    def court_polygon_image(self, margin: float = 0.0) -> np.ndarray:
        """Polígono da quadra projetado na imagem (útil para máscara/visualização)."""
        m = margin
        corners = np.array([[-m, -m], [COURT_W + m, -m], [COURT_W + m, COURT_L + m], [-m, COURT_L + m]])
        return self.to_image(corners)
