"""Identidade simplificada dos 4 jogadores (substitui a re-ID por regras do artigo).

Regra (por frame):
1. Projeta o ponto do pé de cada pessoa na quadra e descarta quem está fora (público, árbitro).
2. Separa por lado da rede: "far" (y < 10) e "near" (y >= 10).
3. Em cada lado mantém até 2 pessoas (maior confiança de detecção x altura da bbox).
4. Com 2 pessoas no lado: menor x = "left", maior x = "right". Com 1: x < 5 -> "left".

Limitação conhecida: se os parceiros cruzam (troca esquerda/direita durante o rally),
o slot acompanha a POSIÇÃO e não a PESSOA. Para avaliar por pessoa, use
`smooth_slots_by_track` (voto de slot por track_id ao longo da janela) ou um mapeamento
pessoa->lado por game no adapter de anotações. Ver docs/SPEC.md §5.2.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from .court import COURT_W, CourtModel
from .types import FrameObs, HitWindow, PlayerObs


def assign_slots_frame(players: list[PlayerObs], court: CourtModel, margin: float = 0.5) -> list[PlayerObs]:
    by_side: dict[str, list[PlayerObs]] = defaultdict(list)
    for p in players:
        cxy = court.to_court(p.foot_point())[0]
        if not court.in_court(cxy, margin):
            continue
        p.court_xy = (float(cxy[0]), float(cxy[1]))
        by_side[court.side(cxy)].append(p)

    kept: list[PlayerObs] = []
    for side, ps in by_side.items():
        ps = sorted(ps, key=lambda q: q.det_conf * q.height, reverse=True)[:2]
        if len(ps) == 2:
            ps.sort(key=lambda q: q.court_xy[0])
            ps[0].slot, ps[1].slot = f"{side}_left", f"{side}_right"
        else:
            ps[0].slot = f"{side}_left" if ps[0].court_xy[0] < COURT_W / 2 else f"{side}_right"
        kept.extend(ps)
    return kept


def smooth_slots_by_track(frames: list[FrameObs]) -> None:
    """Dá a cada track_id o slot mais frequente na janela (estabiliza trocas pontuais)."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for f in frames:
        for p in f.players:
            if p.slot:
                votes[p.track_id][p.slot] += 1
    final = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    for f in frames:
        for p in f.players:
            if p.track_id in final:
                p.slot = final[p.track_id]


def assign_slots_window(window: HitWindow, court: CourtModel, margin: float = 0.5,
                        smooth: bool = True) -> HitWindow:
    for f in window.frames:
        f.players = assign_slots_frame(f.players, court, margin)
    if smooth:
        smooth_slots_by_track(window.frames)
    return window
