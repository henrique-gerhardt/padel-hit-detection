"""Atribuição golpe -> jogador (núcleo do experimento replicado).

Método principal ("wrist_vote", inspirado em Decorte et al., CVPRW 2024):
    Para cada frame da janela centrada no golpe em que a bola foi observada,
    cada jogador recebe um voto ponderado pela distância entre o punho mais próximo e a bola:

        d_p,f  = min_{w in punhos(p)} || w - b_f ||             (pixels)
        s_p,f  = exp( -0.5 * (d_p,f / (sigma_rel * h_p,f))^2 )  (h = altura da bbox -> invariância de escala)
        S_p    = sum_f  tw_f * s_p,f                            (tw = peso temporal opcional)
        jogador = argmax_p S_p

Fallbacks (sem bola na janela): pico de velocidade do punho normalizada pela altura ("wrist_speed").
Baseline de ablação ("bbox_center"): jogador cujo centro da bbox está mais perto da bola no frame do golpe.
Variante "paper_eq1": peso da eq. (1) de Decorte et al., w_i = 1 - (d_i - d_min + 0.1) / (d_max - d_min),
calculado por frame entre os jogadores presentes; empate -> menor distância somada.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .types import WRISTS, FrameObs, HitWindow, team_of


@dataclass
class AttributionConfig:
    method: str = "wrist_vote"  # "wrist_vote" | "bbox_center" | "paper_eq1"
    window_frames: int = 12  # 500 ms a 25 fps
    sigma_rel: float = 0.5  # largura do kernel, em alturas de jogador
    temporal_sigma: float | None = None  # em frames; None = peso uniforme
    min_kp_conf: float = 0.3
    min_ball_conf: float = 0.3
    use_wrist_speed_fallback: bool = True
    min_wrist_speed: float = 0.05  # alturas/frame; abaixo disso ninguém "balançou" -> sem previsão


@dataclass
class AttributionResult:
    hit_id: str
    pred_slot: str | None
    method_used: str  # "wrist_vote" | "bbox_center" | "wrist_speed" | "none"
    n_ball_frames: int
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def pred_team(self) -> str | None:
        return team_of(self.pred_slot)


def window_frames(frames: list[FrameObs], center: int, n: int) -> list[FrameObs]:
    """Frames em [center - n//2, center + n - n//2): exatamente n frames quando disponíveis."""
    lo, hi = center - n // 2, center + (n - n // 2)
    return [f for f in frames if lo <= f.frame_idx < hi]


def _ball_ok(f: FrameObs, cfg: AttributionConfig) -> bool:
    return f.ball is not None and f.ball.conf >= cfg.min_ball_conf


def _wrist_vote(frames: list[FrameObs], center: int, cfg: AttributionConfig) -> tuple[dict[str, float], int]:
    scores: dict[str, float] = defaultdict(float)
    n_ball = 0
    for f in frames:
        if not _ball_ok(f, cfg):
            continue
        n_ball += 1
        b = np.asarray(f.ball.xy, dtype=float)
        tw = 1.0
        if cfg.temporal_sigma:
            tw = float(np.exp(-0.5 * ((f.frame_idx - center) / cfg.temporal_sigma) ** 2))
        for p in f.players:
            if p.slot is None:
                continue
            w = p.wrists(cfg.min_kp_conf)
            if len(w) == 0:
                continue
            d = float(np.min(np.linalg.norm(w - b, axis=1)))
            scores[p.slot] += tw * float(np.exp(-0.5 * (d / (cfg.sigma_rel * p.height)) ** 2))
    return dict(scores), n_ball


def _bbox_center(frames: list[FrameObs], center: int, cfg: AttributionConfig) -> tuple[dict[str, float], int]:
    """Baseline: frame com bola mais próximo do instante do golpe; vence a bbox mais próxima."""
    with_ball = [f for f in frames if _ball_ok(f, cfg) and f.players]
    if not with_ball:
        return {}, 0
    f = min(with_ball, key=lambda fr: abs(fr.frame_idx - center))
    b = np.asarray(f.ball.xy, dtype=float)
    scores = {p.slot: -float(np.linalg.norm(p.bbox_center - b)) / p.height for p in f.players if p.slot}
    return scores, len(with_ball)


def _paper_eq1(frames: list[FrameObs], center: int, cfg: AttributionConfig) -> tuple[dict[str, float], int]:
    """Eq. (1) do artigo. Ponto do jogador: punho mais próximo da bola; sem punho, centro da bbox (§5.3)."""
    scores: dict[str, float] = defaultdict(float)
    dist_sum: dict[str, float] = defaultdict(float)
    n_ball = 0
    for f in frames:
        if not _ball_ok(f, cfg):
            continue
        b = np.asarray(f.ball.xy, dtype=float)
        d: dict[str, float] = {}
        for p in f.players:
            if p.slot is None:
                continue
            w = p.wrists(cfg.min_kp_conf)
            pts = w if len(w) else np.asarray([p.bbox_center], dtype=float)
            d[p.slot] = float(np.min(np.linalg.norm(pts - b, axis=1)))
        if not d:
            continue
        n_ball += 1
        dmin, dmax = min(d.values()), max(d.values())
        for slot, di in d.items():
            # um só jogador no frame: a eq. (1) não se aplica; ele recebe o voto inteiro
            scores[slot] += 1.0 if dmax == dmin else 1.0 - (di - dmin + 0.1) / (dmax - dmin)
            dist_sum[slot] += di
    # desempate pela menor distância somada (perturbação muito menor que qualquer diferença de peso)
    return {s: v - 1e-9 * dist_sum[s] for s, v in scores.items()}, n_ball


def _wrist_speed(frames: list[FrameObs], cfg: AttributionConfig) -> dict[str, float]:
    """Pico da velocidade do punho (px/frame / altura) por slot: proxy do swing da raquete."""
    tracks: dict[str, list[tuple[int, np.ndarray, float]]] = defaultdict(list)
    for f in sorted(frames, key=lambda fr: fr.frame_idx):
        for p in f.players:
            if p.slot is None:
                continue
            for k_idx, k in enumerate(WRISTS):
                if p.kp_conf[k] >= cfg.min_kp_conf:
                    tracks[f"{p.slot}|{k_idx}"].append((f.frame_idx, p.keypoints[k], p.height))
    scores: dict[str, float] = defaultdict(float)
    for key, seq in tracks.items():
        slot = key.split("|")[0]
        for (f0, x0, h0), (f1, x1, h1) in zip(seq, seq[1:]):
            dt = f1 - f0
            if dt <= 0 or dt > 3:
                continue
            v = float(np.linalg.norm(x1 - x0)) / dt / ((h0 + h1) / 2)
            scores[slot] = max(scores[slot], v)
    return dict(scores)


def attribute_hit(window: HitWindow, cfg: AttributionConfig | None = None) -> AttributionResult:
    cfg = cfg or AttributionConfig()
    center = window.hit.frame
    frames = window_frames(window.frames, center, cfg.window_frames)

    if cfg.method == "wrist_vote":
        scores, n_ball = _wrist_vote(frames, center, cfg)
    elif cfg.method == "bbox_center":
        scores, n_ball = _bbox_center(frames, center, cfg)
    elif cfg.method == "paper_eq1":
        scores, n_ball = _paper_eq1(frames, center, cfg)
    else:
        raise ValueError(f"Método desconhecido: {cfg.method}")

    if scores and max(scores.values()) > (0.0 if cfg.method == "wrist_vote" else -np.inf):
        best = max(scores, key=scores.get)
        return AttributionResult(window.hit.hit_id, best, cfg.method, n_ball, scores)

    if cfg.use_wrist_speed_fallback:
        sp = _wrist_speed(frames, cfg)
        if sp and max(sp.values()) >= cfg.min_wrist_speed:
            best = max(sp, key=sp.get)
            return AttributionResult(window.hit.hit_id, best, "wrist_speed", n_ball, sp)

    return AttributionResult(window.hit.hit_id, None, "none", n_ball, {})
