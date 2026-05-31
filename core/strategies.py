"""Prompt muhendisligi stratejileri.

Her strateji, bir gorevi (system, user) mesaj ciftine donusturur. Stratejiler
literaturdeki temel tekniklere karsilik gelir: Zero-shot, Few-shot,
Chain-of-Thought (Wei ve ark., 2022), ReAct (Yao ve ark., 2022),
Tree-of-Thoughts (Yao ve ark., 2023) ve Meta-prompting.

Gorev sozlugu su alanlari icerir:
    question : cozulecek soru/talimat (zorunlu)
    examples : few-shot icin giris-cikis ornekleri listesi (opsiyonel)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Tum stratejiler icin ortak temel sistem talimati.
BASE_SYSTEM = "Sen yardimci ve dikkatli bir asistansin. Sorulari dogru yanitla."


def zero_shot(task: dict) -> tuple[str, str]:
    # En sade temel cizgi (baseline): yalnizca soruyu ver.
    return BASE_SYSTEM, task["question"]


def few_shot(task: dict) -> tuple[str, str]:
    # Soru oncesinde birkac cozulmus ornek gostererek baglam ici ogrenme.
    examples = task.get("examples", [])
    blocks = [f"Soru: {ex['input']}\nCevap: {ex['output']}" for ex in examples]
    prompt = "\n\n".join(blocks + [f"Soru: {task['question']}\nCevap:"])
    return BASE_SYSTEM, prompt


def chain_of_thought(task: dict) -> tuple[str, str]:
    # Modelden cevaptan once adim adim muhakeme istenir (Wei ve ark., 2022).
    system = BASE_SYSTEM + " Cevaptan once adim adim dusun."
    user = task["question"] + "\n\nAdim adim dusunerek coz, sonra son cevabi yaz."
    return system, user


def react(task: dict) -> tuple[str, str]:
    # Dusun-Eylem-Gozlem dongusunu taklit eden yapilandirilmis muhakeme.
    system = BASE_SYSTEM + (
        " Su donguyu izle: Dusunce (durumu degerlendir), Eylem (bir adim at), "
        "Gozlem (sonucu yorumla). Gerekirse tekrarla, sonunda 'Cevap:' ile bitir."
    )
    return system, task["question"]


def tree_of_thoughts(task: dict) -> tuple[str, str]:
    # Birden cok cozum yolu uretip kiyaslama (Yao ve ark., 2023); tek cagrida
    # basitlestirilmis hali.
    system = BASE_SYSTEM + (
        " Soruyu cozmek icin 3 farkli yaklasim uret, her birini kisaca degerlendir, "
        "en guclu olani sec ve son cevabi 'Cevap:' ile ver."
    )
    return system, task["question"]


def meta_prompting(task: dict) -> tuple[str, str]:
    # Modele once bu gorev icin iyi bir talimat yazdirip sonra onu uygulatma.
    user = (
        "Once bu gorevi en iyi cozecek talimati kendin tasarla, sonra o talimati "
        f"uygulayarak coz.\n\nGorev: {task['question']}"
    )
    return BASE_SYSTEM, user


@dataclass
class Strategy:
    """Bir prompt stratejisinin kimligi, gorunen adi ve uretici fonksiyonu."""

    key: str
    name: str
    build: Callable[[dict], tuple[str, str]]


# Arayuz ve degerlendirme motorunun uzerinde donecegi strateji listesi.
STRATEGIES = [
    Strategy("zero_shot", "Zero-shot", zero_shot),
    Strategy("few_shot", "Few-shot", few_shot),
    Strategy("cot", "Chain-of-Thought", chain_of_thought),
    Strategy("react", "ReAct", react),
    Strategy("tot", "Tree-of-Thoughts", tree_of_thoughts),
    Strategy("meta", "Meta-prompting", meta_prompting),
]
