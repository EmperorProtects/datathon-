"""Таймлайн атомарных событий + отладочный вывод (timeline.json, PNG-диаграмма)."""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.timeline.events import build_timeline
from pipeline.types import Event

__all__ = ["build_timeline", "save_timeline"]

INTERVAL_TYPES_ORDER = [
    "cashier_present", "cashier_at_desk", "cashier_left_zone", "cashier_hands_visible", "client_at_window",
    "multiple_clients", "collector_present", "outsider_in_staff_zone", "object_on_desk", "object_unattended",
    "hold_cash", "recount_visible", "cash_at_counter_machine", "cash_handover", "cash_near_body", "cash_near_bag",
    "cash_in_safe", "drinking",
]


def _lane(e: Event) -> str:
    parts = [e.type]
    if e.args.get("cls"):
        parts.append(e.args["cls"])
    if e.actor is not None:
        parts.append(f"{e.role or '?'}#{e.actor}")
    return ":".join(parts)


def save_timeline(events: list[Event], duration: float, json_path: Path, png_path: Path | None = None,
                  marks: list[tuple[float, str]] | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([e.to_dict() for e in events], ensure_ascii=False, indent=1), encoding="utf-8")
    if png_path is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = {k: i for i, k in enumerate(INTERVAL_TYPES_ORDER)}
    lanes = sorted({_lane(e) for e in events}, key=lambda s: (order.get(s.split(":")[0], 99), s))
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.28 * len(lanes) + 1)))
    for y, lane in enumerate(lanes):
        for e in events:
            if _lane(e) != lane:
                continue
            if e.t_end > e.t_start:
                ax.barh(y, e.t_end - e.t_start, left=e.t_start, height=0.6, color="#4c8dff", alpha=0.35 + 0.6 * e.conf)
            else:
                ax.plot(e.t_start, y, marker="v", color="#c93c37", markersize=6)
    for t, label in marks or []:
        ax.axvline(t, color="#c93c37", lw=1.2, ls="--")
        ax.text(t, len(lanes) - 0.4, label, color="#c93c37", fontsize=7, rotation=90, va="top")
    ax.set_yticks(range(len(lanes)), lanes, fontsize=7)
    ax.set_xlim(0, duration)
    ax.set_xlabel("секунды")
    ax.invert_yaxis()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=90)
    plt.close(fig)
