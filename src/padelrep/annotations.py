"""Leitura de anotações.

O restante do código só conhece o FORMATO CANÔNICO (CSV), para isolar o formato do
dataset do artigo:

    hit_id,video_id,frame,gt_slot,rally_id
    r001_h03,match_A,15234,near_left,r001
    r001_h04,match_A,15270,,r001          <- golpe sem jogador anotado (fica fora da métrica)

`gt_slot` em {far_left, far_right, near_left, near_right} (relativo à câmera, ver types.py).

Adapter do dataset original: `convert_decorte_to_canonical` (fim deste arquivo). Respostas às
perguntas sobre o formato (Q1–Q5) em docs/SPEC.md §3.3.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .types import SLOTS, Hit

CANONICAL_FIELDS = ["hit_id", "video_id", "frame", "gt_slot", "rally_id"]


def load_canonical_csv(path: str | Path, only_labeled: bool = False) -> list[Hit]:
    hits: list[Hit] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            slot = (row.get("gt_slot") or "").strip() or None
            if slot is not None and slot not in SLOTS:
                raise ValueError(f"gt_slot inválido '{slot}' em {row}")
            if only_labeled and slot is None:
                continue
            hits.append(Hit(
                hit_id=row["hit_id"],
                video_id=row["video_id"],
                frame=int(float(row["frame"])),
                gt_slot=slot,
                rally_id=(row.get("rally_id") or None),
            ))
    return hits


def save_canonical_csv(hits: list[Hit], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CANONICAL_FIELDS)
        w.writeheader()
        for h in hits:
            w.writerow({"hit_id": h.hit_id, "video_id": h.video_id, "frame": h.frame,
                        "gt_slot": h.gt_slot or "", "rally_id": h.rally_id or ""})


# ---------------------------------------------------------------------------
# Adapter do dataset de Decorte et al. (CVsports 2024)
# ---------------------------------------------------------------------------
# Estrutura baixada do servidor imec (ver docs/SPEC.md §3.3):
#   rallies/<DATA>_<LOCAL>_<NN>.mp4           um vídeo por rally (99)
#   metadata/hits.csv                          filename,start,end,class,class_id   (2.377 golpes, segundos, áudio)
#   metadata/hit_assignments.xlsx              <idx>,video,timestamp,player          (319 golpes com jogador)
# `timestamp` (mm:ss.mmm) é o frame mais próximo em que o golpe é visível; `player` segue o esquema
# do artigo (§5.2): número pela posição no início do rally, vista da câmera.

DECORTE_PLAYER_TO_SLOT = {
    "t1p1": "far_left",    # top-left
    "t1p2": "far_right",   # top-right
    "t2p1": "near_left",   # bottom-left
    "t2p2": "near_right",  # bottom-right
}

# Distância máxima (s) entre o instante do jogador anotado e o onset de áudio para considerar
# que os dois registros são o mesmo golpe.
MATCH_TOL_S = 0.5


def parse_mmss(value) -> float:
    """'01:02.360' -> 62.36 s. Aceita também número (segundos)."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        m, s = text.split(":", 1)
        return int(m) * 60 + float(s)
    return float(text)


def read_decorte_hits(path: str | Path) -> list[dict]:
    """metadata/hits.csv -> [{video_id, start, end}] (segundos relativos ao início do rally)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"video_id": Path(row["filename"]).stem,
                         "start": float(row["start"]), "end": float(row["end"])})
    return rows


def read_decorte_assignments(path: str | Path) -> list[dict]:
    """metadata/hit_assignments.xlsx -> [{idx, video_id, t, slot}]."""
    from openpyxl import load_workbook  # import tardio: só o adapter precisa

    ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(it)]
    col = {name: header.index(name) for name in ("video", "timestamp", "player")}
    idx_col = 0 if header[0] in ("", "None") else None
    rows = []
    for n, r in enumerate(it):
        if r[col["video"]] is None:
            continue
        player = str(r[col["player"]]).strip()
        if player not in DECORTE_PLAYER_TO_SLOT:
            raise ValueError(f"Jogador desconhecido '{player}' na linha {n + 2} de {path}")
        rows.append({
            "idx": int(r[idx_col]) if idx_col is not None else n,
            "video_id": Path(str(r[col["video"]]).strip()).stem,
            "t": parse_mmss(r[col["timestamp"]]),
            "slot": DECORTE_PLAYER_TO_SLOT[player],
        })
    return rows


def match_assignments_to_hits(assign: list[dict], hits: list[dict], tol_s: float = MATCH_TOL_S) -> dict[int, int]:
    """Casa cada golpe com jogador anotado a um onset de áudio do mesmo vídeo (1 para 1).

    Os dois registros estão em ordem temporal dentro do rally, então o casamento preserva a ordem:
    alinhamento ótimo por programação dinâmica (como distância de edição), com custo |t - start|
    por par (só pares com |t - start| <= tol_s) e custo tol_s por registro deixado sem par.
    Retorna {índice em `assign`: índice em `hits`}.
    """
    out: dict[int, int] = {}
    for vid in {a["video_id"] for a in assign}:
        A = sorted((i for i, a in enumerate(assign) if a["video_id"] == vid), key=lambda i: assign[i]["t"])
        B = sorted((i for i, h in enumerate(hits) if h["video_id"] == vid), key=lambda i: hits[i]["start"])
        n, m = len(A), len(B)
        INF = float("inf")
        cost = [[INF] * (m + 1) for _ in range(n + 1)]
        move = [[None] * (m + 1) for _ in range(n + 1)]
        cost[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                c = cost[i][j]
                if c == INF:
                    continue
                if i < n and c + tol_s < cost[i + 1][j]:
                    cost[i + 1][j], move[i + 1][j] = c + tol_s, "skip_a"
                if j < m and c + tol_s < cost[i][j + 1]:
                    cost[i][j + 1], move[i][j + 1] = c + tol_s, "skip_h"
                if i < n and j < m:
                    d = abs(assign[A[i]]["t"] - hits[B[j]]["start"])
                    if d <= tol_s and c + d < cost[i + 1][j + 1]:
                        cost[i + 1][j + 1], move[i + 1][j + 1] = c + d, "match"
        i, j = n, m
        while i or j:
            mv = move[i][j]
            if mv == "match":
                out[A[i - 1]] = B[j - 1]
                i, j = i - 1, j - 1
            elif mv == "skip_a":
                i -= 1
            else:
                j -= 1
    return out


def video_fps_table(videos_dir: str | Path, video_ids, ext: str = ".mp4") -> dict[str, float]:
    """FPS real de cada vídeo, lido do arquivo (cv2)."""
    import cv2

    table = {}
    for vid in sorted(set(video_ids)):
        path = Path(videos_dir) / f"{vid}{ext}"
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        if not fps or fps <= 0:
            raise FileNotFoundError(f"Não consegui ler o FPS de {path}")
        table[vid] = float(fps)
    return table


def convert_decorte_to_canonical(raw_dir: str | Path, out_csv: str | Path,
                                 fps: float | dict[str, float] | None = None,
                                 tol_s: float = MATCH_TOL_S) -> dict:
    """Converte as anotações originais do dataset de Decorte et al. para o CSV canônico.

    - Golpes de `hits.csv` sem jogador: frame = round(onset * fps).
    - Golpes de `hit_assignments.xlsx`: frame = round(timestamp * fps) (frame visível do contato),
      slot pelo esquema t1p1..t2p2. Se casar com um onset de áudio (|Δt| <= tol_s), substitui aquele
      golpe; senão entra como linha extra (hit_id com sufixo `_x`).
    - `fps`: None = FPS real de cada vídeo em <raw_dir>/rallies; número ou dict força o valor.

    Retorna um resumo com as contagens (para conferir com o artigo: 2.377 golpes, 319 com jogador).
    """
    raw = Path(raw_dir)
    hits = read_decorte_hits(raw / "metadata" / "hits.csv")
    assign = read_decorte_assignments(raw / "metadata" / "hit_assignments.xlsx")
    vids = {h["video_id"] for h in hits} | {a["video_id"] for a in assign}
    if fps is None:
        fps_of = video_fps_table(raw / "rallies", vids)
    elif isinstance(fps, dict):
        fps_of = dict(fps)
    else:
        fps_of = {v: float(fps) for v in vids}

    match = match_assignments_to_hits(assign, hits, tol_s)
    by_hit = {hi: ai for ai, hi in match.items()}

    # numeração estável dos golpes de áudio por vídeo, em ordem de onset
    order: dict[str, list[int]] = {}
    for hi, h in enumerate(hits):
        order.setdefault(h["video_id"], []).append(hi)
    hit_ids = {}
    for vid, idxs in order.items():
        for k, hi in enumerate(sorted(idxs, key=lambda i: hits[i]["start"])):
            hit_ids[hi] = f"{vid}_h{k:03d}"

    out: list[Hit] = []
    for hi, h in enumerate(hits):
        vid = h["video_id"]
        if hi in by_hit:
            a = assign[by_hit[hi]]
            out.append(Hit(hit_ids[hi], vid, int(round(a["t"] * fps_of[vid])), a["slot"], vid))
        else:
            out.append(Hit(hit_ids[hi], vid, int(round(h["start"] * fps_of[vid])), None, vid))
    unmatched = [ai for ai in range(len(assign)) if ai not in match]
    for ai in unmatched:
        a = assign[ai]
        vid = a["video_id"]
        out.append(Hit(f"{vid}_a{a['idx']:03d}_x", vid, int(round(a["t"] * fps_of[vid])), a["slot"], vid))

    out.sort(key=lambda x: (x.video_id, x.frame, x.hit_id))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    save_canonical_csv(out, out_csv)
    return {
        "n_audio_hits": len(hits),
        "n_assignments": len(assign),
        "n_matched": len(match),
        "n_unmatched_assignments": len(unmatched),
        "n_rows": len(out),
        "n_labeled": sum(h.gt_slot is not None for h in out),
        "n_videos": len(vids),
        "n_videos_labeled": len({a["video_id"] for a in assign}),
        "fps": sorted(set(fps_of.values())),
    }
