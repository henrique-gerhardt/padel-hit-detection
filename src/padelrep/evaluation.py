"""Métricas de atribuição e comparação com o artigo."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .attribution import AttributionResult
from .types import SLOTS, Hit

# Decorte et al., CVPRW 2024 — subconjunto de 319 golpes com jogador anotado.
PAPER_RESULTS = {"player_acc": 0.8370, "team_acc": 0.8683, "n_hits": 319}


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(correct) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    accs = correct[idx].mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def evaluate(results: list[AttributionResult], hits: list[Hit]) -> dict:
    """Golpes sem previsão (method_used == 'none') contam como ERRO (protocolo conservador)."""
    gt = {h.hit_id: h for h in hits if h.gt_slot is not None}
    pairs = [(r, gt[r.hit_id]) for r in results if r.hit_id in gt]
    if not pairs:
        raise ValueError("Nenhum golpe com ground truth casou com as previsões.")

    player_ok = np.array([r.pred_slot == h.gt_slot for r, h in pairs], dtype=float)
    team_ok = np.array([r.pred_team is not None and r.pred_team == h.gt_team for r, h in pairs], dtype=float)

    by_method: dict[str, dict] = {}
    for m in sorted({r.method_used for r, _ in pairs}):
        mask = np.array([r.method_used == m for r, _ in pairs])
        by_method[m] = {
            "n": int(mask.sum()),
            "player_acc": float(player_ok[mask].mean()),
            "team_acc": float(team_ok[mask].mean()),
        }

    labels = list(SLOTS) + ["none"]
    confusion = {g: Counter() for g in SLOTS}
    for r, h in pairs:
        confusion[h.gt_slot][r.pred_slot or "none"] += 1

    return {
        "n_hits": len(pairs),
        "coverage": float(np.mean([r.pred_slot is not None for r, _ in pairs])),
        "player_acc": float(player_ok.mean()),
        "player_acc_ci95": _bootstrap_ci(player_ok),
        "team_acc": float(team_ok.mean()),
        "team_acc_ci95": _bootstrap_ci(team_ok, seed=1),
        "by_method": by_method,
        "confusion": {g: {p: int(confusion[g][p]) for p in labels} for g in SLOTS},
        "paper": PAPER_RESULTS,
    }


def format_report(metrics: dict, title: str = "Resultados") -> str:
    p = metrics["paper"]
    lo, hi = metrics["player_acc_ci95"]
    tlo, thi = metrics["team_acc_ci95"]
    lines = [
        f"# {title}",
        "",
        "| Métrica | Este trabalho | Artigo (Decorte et al., 2024) |",
        "|---|---|---|",
        f"| Golpes avaliados | {metrics['n_hits']} | {p['n_hits']} |",
        f"| Acurácia por jogador | {metrics['player_acc']:.2%} (IC95% {lo:.1%}–{hi:.1%}) | {p['player_acc']:.2%} |",
        f"| Acurácia por dupla | {metrics['team_acc']:.2%} (IC95% {tlo:.1%}–{thi:.1%}) | {p['team_acc']:.2%} |",
        f"| Cobertura (golpes com previsão) | {metrics['coverage']:.1%} | — |",
        "",
        "## Por método efetivamente usado",
        "",
        "| Método | n | Acc. jogador | Acc. dupla |",
        "|---|---|---|---|",
    ]
    for m, v in metrics["by_method"].items():
        lines.append(f"| {m} | {v['n']} | {v['player_acc']:.1%} | {v['team_acc']:.1%} |")
    labels = list(SLOTS) + ["none"]
    lines += ["", "## Matriz de confusão (linhas = GT, colunas = previsto)", "",
              "| GT \\ Pred | " + " | ".join(labels) + " |",
              "|---" * (len(labels) + 1) + "|"]
    for g in SLOTS:
        lines.append(f"| {g} | " + " | ".join(str(metrics["confusion"][g][x]) for x in labels) + " |")
    return "\n".join(lines) + "\n"
