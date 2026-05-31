"""Gorev veri setlerini JSON dosyalarindan yukler.

Her veri seti `data/tasks/` altinda bir JSON dosyasidir ve su yapidadir:
    name      : gorunen ad
    task_type : "arithmetic" (sayisal) | "classification" (etiket)
    examples  : few-shot icin ortak giris-cikis ornekleri
    tasks     : {question, answer} sozluklerinin listesi
"""
from __future__ import annotations

import json
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent.parent / "data" / "tasks"


def list_datasets() -> list[str]:
    """Mevcut veri seti adlarini (dosya adi, uzantisiz) dondurur."""
    return sorted(p.stem for p in TASKS_DIR.glob("*.json"))


def load_dataset(name: str) -> dict:
    """Adi verilen veri setini yukler ve ornekleri her goreve ilistirir."""
    path = TASKS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    # Few-shot stratejisinin kullanmasi icin veri seti duzeyindeki ornekleri
    # her gorev sozlugune kopyala (gorev kendi orneklerini tanimlamamissa).
    examples = data.get("examples", [])
    for task in data["tasks"]:
        task.setdefault("examples", examples)
    return data
