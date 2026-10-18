"""Линтер локальности: в pipeline/ и app/worker/ нет сетевых клиентов и облачных AI SDK."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(
    r"^\s*(import|from)\s+(requests|httpx|aiohttp|urllib\.request|urllib3|openai|anthropic|"
    r"google\.generativeai|google\.genai|replicate|boto3)\b",
    re.MULTILINE,
)


def test_no_network_clients_in_inference_code():
    offenders = []
    for d in ("pipeline", "app/worker"):
        for py in (ROOT / d).rglob("*.py"):
            for m in FORBIDDEN.finditer(py.read_text(encoding="utf-8")):
                offenders.append(f"{py.relative_to(ROOT)}: {m.group(0).strip()}")
    assert not offenders, "network imports found:\n" + "\n".join(offenders)
