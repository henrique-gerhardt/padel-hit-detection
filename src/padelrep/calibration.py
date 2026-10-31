"""Calibração automática da quadra: pontos de referência a partir das linhas da imagem.

1. Imagem de fundo: mediana de frames de vários rallies da mesma câmera (os jogadores somem).
2. Retas ajustadas às linhas de saque e à linha central (pixels brancos) e às bordas do piso azul, dentro de faixas
   de busca definidas por câmera. As laterais são ajustadas em dois trechos (longe e perto), porque a lente curva
   levemente as bordas. As linhas pintadas parecem continuar além do piso porque refletem no vidro; por isso as
   pontas das linhas de saque vêm da interseção com a borda do piso.
3. Interseções -> 10 pontos nomeados (4 cantos, 4 pontas das linhas de saque, 2 "T") -> homografia (RANSAC).
"""
from __future__ import annotations

import cv2
import numpy as np

from .court import COURT_L, COURT_W, NET_Y, REFERENCE_POINTS, CourtModel


def median_background(videos: list[str], per_video: int = 8) -> np.ndarray:
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in np.linspace(0, n - 1, per_video).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, im = cap.read()
            if ok:
                frames.append(im)
        cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def white_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 70) & (hsv[..., 2] > 170)


def blue_mask(img: np.ndarray, h=(104, 123), s_min=120, v_min=80) -> np.ndarray:
    """Piso azul (H do OpenCV ~104-123), saturado. `v_min` separa o piso do piso visto pelo vidro."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= h[0]) & (hsv[..., 0] <= h[1]) & (hsv[..., 1] > s_min) & (hsv[..., 2] > v_min)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def fit_line(pts: np.ndarray) -> np.ndarray:
    """Reta robusta (Huber) -> (a, b, c) com a x + b y + c = 0, (a, b) unitário."""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    a, b = vy, -vx
    return np.array([a, b, -(a * x0 + b * y0)], dtype=float)


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def horizontal_white(mask, x0, x1, y0, y1):
    """Linha branca quase horizontal: centróide vertical dos pixels brancos por coluna."""
    pts = []
    for x in range(x0, x1):
        ys = np.nonzero(mask[y0:y1, x])[0]
        if 1 <= len(ys) <= 12:
            pts.append((x, y0 + ys.mean()))
    return np.array(pts)


def vertical_white(mask, x0, x1, y0, y1):
    pts = []
    for y in range(y0, y1):
        xs = np.nonzero(mask[y, x0:x1])[0]
        if 1 <= len(xs) <= 12:
            pts.append((x0 + xs.mean(), y))
    return np.array(pts)


def floor_edge_rows(blue, x0, x1, y0, y1, top: bool):
    """Borda horizontal do piso: primeira (top) ou última linha azul de cada coluna na faixa."""
    pts = []
    for x in range(x0, x1):
        ys = np.nonzero(blue[y0:y1, x])[0]
        if len(ys):
            pts.append((x, y0 + (ys.min() if top else ys.max())))
    return np.array(pts)


def floor_edge_cols(blue, x0, x1, y0, y1, left: bool):
    """Borda lateral do piso: primeira (left) ou última coluna azul de cada linha na faixa."""
    pts = []
    for y in range(y0, y1):
        xs = np.nonzero(blue[y, x0:x1])[0]
        if len(xs):
            pts.append((x0 + (xs.min() if left else xs.max()), y))
    return np.array(pts)


def band_list(band) -> list[list[int]]:
    """Uma faixa [x0, x1, y0, y1] ou uma lista de faixas."""
    return band if isinstance(band[0], (list, tuple)) else [band]


def suggest(img: np.ndarray, bands: dict) -> tuple[dict[str, list[float]], dict[str, np.ndarray], dict]:
    bands = dict(bands)
    mask_cfg = bands.pop("blue_mask", {})
    W, B = white_mask(img), blue_mask(img, **mask_cfg)
    # fundo longe: o vidro do canto escurece o piso; limiar próprio opcional
    BF = blue_mask(img, **{**mask_cfg, **bands.pop("blue_mask_far", {})})
    finders = {
        "far_service": lambda b: horizontal_white(W, *b),
        "near_service": lambda b: horizontal_white(W, *b),
        "center": lambda b: vertical_white(W, *b),
        "far_base": lambda b: floor_edge_rows(BF, *b, top=True),
        "near_base": lambda b: floor_edge_rows(B, *b, top=False),
        "left_side_far": lambda b: floor_edge_cols(BF, *b, left=True),
        "right_side_far": lambda b: floor_edge_cols(BF, *b, left=False),
        "left_side_near": lambda b: floor_edge_cols(B, *b, left=True),
        "right_side_near": lambda b: floor_edge_cols(B, *b, left=False),
    }
    samples = {}
    for k, find in finders.items():
        parts = [find(b) for b in band_list(bands[k])]
        samples[k] = np.concatenate([q for q in parts if len(q)])
    lines = {k: fit_line(v) for k, v in samples.items()}
    named = {
        "far_left_corner": ("far_base", "left_side_far"),
        "far_right_corner": ("far_base", "right_side_far"),
        "near_left_corner": ("near_base", "left_side_near"),
        "near_right_corner": ("near_base", "right_side_near"),
        "far_service_left": ("far_service", "left_side_far"),
        "far_service_right": ("far_service", "right_side_far"),
        "far_service_center": ("far_service", "center"),
        "near_service_left": ("near_service", "left_side_near"),
        "near_service_right": ("near_service", "right_side_near"),
        "near_service_center": ("near_service", "center"),
    }
    pts = {n: [round(float(v), 1) for v in intersect(lines[a], lines[b])] for n, (a, b) in named.items()}
    resid = {}
    for k, v in samples.items():
        a, b, c = lines[k]
        d = np.abs(v[:, 0] * a + v[:, 1] * b + c)
        resid[k] = {"n": int(len(v)), "median_px": round(float(np.median(d)), 2)}
    return pts, samples, resid


def reprojection_error(court: CourtModel, pts: dict[str, list[float]]) -> np.ndarray:
    """Erro (m) de cada ponto marcado, projetado na quadra pela homografia ajustada."""
    img = np.array(list(pts.values()))
    gt = np.array([REFERENCE_POINTS[n] for n in pts])
    return np.linalg.norm(court.to_court(img) - gt, axis=1)


def draw_court_overlay(img, court: CourtModel):
    segs = [((0, 0), (COURT_W, 0)), ((COURT_W, 0), (COURT_W, COURT_L)), ((COURT_W, COURT_L), (0, COURT_L)),
            ((0, COURT_L), (0, 0)), ((0, NET_Y), (COURT_W, NET_Y)),
            ((0, 3.05), (COURT_W, 3.05)), ((0, 16.95), (COURT_W, 16.95)),
            ((5, 3.05), (5, 16.95))]
    for a, b in segs:
        pa, pb = court.to_image(np.array([a, b]))
        cv2.line(img, tuple(map(int, pa)), tuple(map(int, pb)), (0, 255, 255), 2)
    return img
