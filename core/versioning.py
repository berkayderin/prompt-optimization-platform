"""Prompt surumleme ve A/B karsilastirma.

Her prompt surumu, metni ve olculen metrikleriyle birlikte JSON olarak saklanir.
Boylece zamanla hangi surumun daha iyi sonuc verdigi izlenebilir ve iki surum
A/B testi mantigiyla karsilastirilabilir.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


def prompt_id(text: str) -> str:
    """Prompt metnine icerik tabanli kararli bir kimlik uretir."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@dataclass
class PromptVersion:
    """Saklanan bir prompt surumu ve ona ait olcumler."""

    pid: str
    strategy_key: str
    strategy_name: str
    accuracy: float
    avg_tokens: float


def save_version(version: PromptVersion, store: str = "versions.json") -> None:
    """Surumu JSON deposunun sonuna ekler (yoksa olusturur)."""
    # Sonuc klasoru yoksa olustur; temiz bir kurulumda veya Docker imajinda
    # (results klasoru imaja dahil edilmez) bu dizin bulunmayabilir.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / store
    records = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
    records.append(asdict(version))
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def compare(a: PromptVersion, b: PromptVersion) -> dict:
    """Iki surumu dogruluk ve token acisindan karsilastirir (A/B testi)."""
    return {
        "kazanan": a.pid if a.accuracy >= b.accuracy else b.pid,
        "dogruluk_farki": round(a.accuracy - b.accuracy, 3),
        "token_farki": round(a.avg_tokens - b.avg_tokens, 1),
    }
