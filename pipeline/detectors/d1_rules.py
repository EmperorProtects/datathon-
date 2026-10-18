"""D1 — правила по событиям таймлайна и визуальным состояниям предметов.

Каждое правило даёт кандидата с наблюдаемыми фактами; тип без норматива в реестре не сообщается.
"""
from __future__ import annotations

from pipeline.types import Candidate, Event

UNATTENDED_TYPES = {
    "cash": "CASH_LEFT_UNATTENDED", "cash_bundle": "CASH_LEFT_UNATTENDED",
    "collector_bag": "CASH_BAG_LEFT_UNATTENDED", "document": "DOCUMENTS_LEFT_UNATTENDED",
    "stamp": "STAMPS_LEFT_UNATTENDED",
}
CLS_RU = {"cash": "купюры", "cash_bundle": "пачка купюр", "collector_bag": "инкассаторская сумка",
          "document": "документы", "stamp": "печать/штамп", "bag": "личная сумка", "cup": "чашка"}


def _dur(e: Event) -> float:
    return e.t_end - e.t_start


def _of(events: list[Event], etype: str, cls: str | None = None) -> list[Event]:
    return [e for e in events if e.type == etype and (cls is None or e.args.get("cls") == cls)]


def _overlaps(e: Event, a: float, b: float) -> bool:
    return e.t_end >= a and e.t_start <= b


def away_intervals(events: list[Event], rc: dict, t0: float) -> list[Event]:
    """Кассир ушёл с рабочего места (не начало ролика, когда он ещё не пришёл)."""
    return [e for e in _of(events, "cashier_left_zone")
            if _dur(e) >= rc["away_min_sec"] and e.t_start > t0 + rc["start_grace_sec"]]


def run_d1(events: list[Event], states: dict[float, dict[str, float]], cfg: dict, t0: float) -> list[Candidate]:
    rc = cfg["rules_engine"]
    out: list[Candidate] = []

    # --- кассир ушёл: что осталось без присмотра ---
    for away in away_intervals(events, rc, t0):
        a, b = away.t_start, away.t_end
        for cls, type_id in UNATTENDED_TYPES.items():
            for e in _of(events, "object_unattended", cls):
                if _overlaps(e, a, b) and _dur(e) >= rc["unattended_min_sec"]:
                    t = max(a, e.t_start)
                    out.append(Candidate(type_id, t, t, e.t_end, 0.6 + 0.4 * e.conf, "D1", [
                        f"в {a:.1f} с кассир уходит с рабочего места",
                        f"{CLS_RU[cls]} остаётся на столе без присмотра {_dur(e):.1f} с"]))
        st = [v for t, v in sorted(states.items()) if a + rc["state_delay_sec"] <= t <= b]
        if st:
            bright = max(s.get("laptop_bright", 0.0) for s in st)
            if bright >= rc["laptop_bright_min"]:
                out.append(Candidate("POWERED_EQUIPMENT_LEFT_UNATTENDED", a, a, b, rc["state_score"], "D1", [
                    f"в {a:.1f} с кассир уходит с рабочего места",
                    "экран ноутбука остаётся включённым (светится) в его отсутствие"]))
            safe = max(s.get("safe_change", 0.0) for s in st)
            if safe >= rc["safe_change_min"]:
                out.append(Candidate("SAFE_LEFT_OPEN", a, a, b, rc["state_score"], "D1", [
                    f"в {a:.1f} с кассир уходит с рабочего места",
                    "вид сейфа изменился относительно момента, когда кассир был на месте (вероятно, дверца открыта)"]))
            door = max(s.get("door_change", 0.0) for s in st)
            if door >= rc["door_change_min"]:
                out.append(Candidate("CASHROOM_DOOR_LEFT_OPEN", a, a, b, rc["state_score"], "D1", [
                    f"в {a:.1f} с кассир уходит с рабочего места",
                    "вид двери помещения изменился (вероятно, дверь осталась открытой)"]))

    # --- посторонний на стороне персонала (через открытую дверь) ---
    for e in _of(events, "outsider_in_staff_zone"):
        if _dur(e) >= rc["outsider_min_sec"]:
            out.append(Candidate("CASHROOM_DOOR_LEFT_OPEN", e.t_start, e.t_start, e.t_end, 0.7, "D1", [
                f"в {e.t_start:.1f} с в зоне персонала находится посторонний человек ({_dur(e):.1f} с)"]))

    # --- несколько клиентов одновременно ---
    for e in _of(events, "multiple_clients"):
        if _dur(e) >= rc["multiple_clients_min_sec"]:
            out.append(Candidate("MULTIPLE_CUSTOMERS_SERVED", e.t_start, e.t_start, e.t_end, 0.75, "D1", [
                f"с {e.t_start:.1f} по {e.t_end:.1f} с у стола одновременно находятся два посетителя"]))

    # --- личные вещи и напитки на рабочем месте ---
    for e in _of(events, "object_on_desk", "bag"):
        if _dur(e) >= rc["desk_item_min_sec"]:
            out.append(Candidate("PERSONAL_ITEMS_ON_CASH_DESK", e.t_start, e.t_start, e.t_end, 0.6 + 0.4 * e.conf, "D1", [
                f"с {e.t_start:.1f} с на рабочем столе кассира лежит личная сумка"]))
    for e in _of(events, "drinking") + _of(events, "object_on_desk", "cup"):
        if _dur(e) >= rc["drinking_min_sec"]:
            what = "кассир подносит чашку к лицу" if e.type == "drinking" else "на рабочем столе стоит чашка с напитком"
            out.append(Candidate("BEVERAGE_CONSUMPTION_AT_WORKSTATION", e.t_start, e.t_start, e.t_end,
                                 0.6 + 0.4 * e.conf, "D1", [f"в {e.t_start:.1f} с {what}"]))

    # --- деньги после ухода клиента ---
    client_left = [e.t_start for e in _of(events, "client_left_window")]
    client_on = _of(events, "client_at_window")
    for e in _of(events, "hold_cash"):
        if e.role != "cashier":
            continue
        for tl in client_left:
            if e.t_end > tl + rc["after_client_min_sec"] and not any(
                    _overlaps(c, max(e.t_start, tl), e.t_end) and c.t_start > tl for c in client_on):
                t = max(e.t_start, tl)
                if e.t_end - t >= rc["after_client_min_sec"]:
                    out.append(Candidate("CASH_COUNT_CONTINUES_WITHOUT_CUSTOMER", t, t, e.t_end, 0.6, "D1", [
                        f"в {tl:.1f} с клиент уходит от стола",
                        f"после этого кассир продолжает работать с купюрами до {e.t_end:.1f} с"]))
                break

    # --- деньги прежнего клиента на столе при появлении нового ---
    for arrived in _of(events, "client_arrived"):
        for cls in ("cash", "cash_bundle"):
            for e in _of(events, "object_on_desk", cls):
                if e.t_start <= arrived.t_start - rc["prev_cash_lead_sec"] and e.t_end >= arrived.t_start + 1.0:
                    out.append(Candidate("PREVIOUS_CUSTOMER_CASH_UNSECURED", arrived.t_start, arrived.t_start,
                                         e.t_end, 0.65, "D1", [
                                             f"с {e.t_start:.1f} с на столе лежат купюры",
                                             f"в {arrived.t_start:.1f} с подходит следующий посетитель, деньги не убраны"]))

    # --- инкассация: сумка без документов; деньги в сейф без пересчёта ---
    docs = _of(events, "object_on_desk", "document")
    for e in _of(events, "collector_present"):
        if _dur(e) >= rc["collector_min_sec"] and not any(_overlaps(d, e.t_start - 2, e.t_end + 2) for d in docs):
            out.append(Candidate("CASH_BAG_ACCEPTED_WITHOUT_VISIBLE_DOCUMENTS", e.t_start, e.t_start, e.t_end, 0.55, "D1", [
                f"с {e.t_start:.1f} с у стола находится инкассатор с сумкой",
                "передача или проверка сопроводительных документов на видео не наблюдается"]))
    recounts = _of(events, "recount_visible")
    for e in _of(events, "cash_in_safe") + [d for d in _of(events, "cash_disappeared") if d.args.get("last_zone") == "safe"]:
        if not any(r.t_end >= e.t_start - rc["recount_lookback_sec"] and r.t_start <= e.t_start for r in recounts):
            out.append(Candidate("INCASSATION_CASH_STORED_WITHOUT_VISIBLE_RECOUNT", e.t_start, e.t_start, e.t_end, 0.5, "D1", [
                f"в {e.t_start:.1f} с купюры помещаются в сейф",
                f"пересчёта в предшествующие {rc['recount_lookback_sec']:.0f} с не наблюдается"]))

    # --- передача денег кассир → клиент ---
    for e in _of(events, "cash_handover"):
        if e.args.get("from_role") == "cashier" and e.args.get("to_role") == "client":
            out.append(Candidate("CASH_PARTIALLY_RETURNED", e.t_start, e.t_start, e.t_end, 0.45, "D1", [
                f"в {e.t_start:.1f} с кассир передаёт купюры клиенту через стол"]))

    # --- деньги исчезают у корпуса кассира ---
    for e in _of(events, "cash_disappeared"):
        if e.args.get("last_zone") == "body" and e.role == "cashier":
            out.append(Candidate("PERSONAL_ITEMS_ON_CASH_DESK", e.t_start, e.t_start, e.t_end, 0.5, "D1", [
                f"в {e.t_start:.1f} с купюры исчезают из вида у корпуса кассира, не попав в сейф"]))
    return out
