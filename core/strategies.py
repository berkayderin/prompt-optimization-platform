"""Prompt mühendisliği stratejileri.

Her strateji, bir görevi (system, user) mesaj çiftine dönüştürür. Stratejiler
literatürdeki temel tekniklere karşılık gelir: Zero-shot, Few-shot,
Chain-of-Thought (Wei ve ark., 2022), ReAct (Yao ve ark., 2022),
Tree-of-Thoughts (Yao ve ark., 2023) ve Meta-prompting.

Görev sözlüğü şu alanları içerir:
    question : çözülecek soru/talimat (zorunlu)
    examples : few-shot için giriş-çıkış örnekleri listesi (opsiyonel)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Tüm stratejiler için ortak temel sistem talimatı.
BASE_SYSTEM = "Sen yardımcı ve dikkatli bir asistansın. Soruları doğru yanıtla."


def zero_shot(task: dict) -> tuple[str, str]:
    # En sade temel çizgi (baseline): yalnızca soruyu ver.
    return BASE_SYSTEM, task["question"]


def few_shot(task: dict) -> tuple[str, str]:
    # Soru öncesinde birkaç çözülmüş örnek göstererek bağlam içi öğrenme.
    examples = task.get("examples", [])
    blocks = [f"Soru: {ex['input']}\nCevap: {ex['output']}" for ex in examples]
    prompt = "\n\n".join(blocks + [f"Soru: {task['question']}\nCevap:"])
    return BASE_SYSTEM, prompt


def chain_of_thought(task: dict) -> tuple[str, str]:
    # Modelden cevaptan önce adım adım muhakeme istenir (Wei ve ark., 2022).
    system = BASE_SYSTEM + " Cevaptan önce adım adım düşün."
    user = task["question"] + "\n\nAdım adım düşünerek çöz, sonra son cevabı yaz."
    return system, user


def react(task: dict) -> tuple[str, str]:
    # Düşün-Eylem-Gözlem döngüsünü taklit eden yapılandırılmış muhakeme.
    system = BASE_SYSTEM + (
        " Şu döngüyü izle: Düşünce (durumu değerlendir), Eylem (bir adım at), "
        "Gözlem (sonucu yorumla). Gerekirse tekrarla, sonunda 'Cevap:' ile bitir."
    )
    return system, task["question"]


def tree_of_thoughts(task: dict) -> tuple[str, str]:
    # Birden çok çözüm yolu üretip kıyaslama (Yao ve ark., 2023); tek çağrıda
    # basitleştirilmiş hali.
    system = BASE_SYSTEM + (
        " Soruyu çözmek için 3 farklı yaklaşım üret, her birini kısaca değerlendir, "
        "en güçlü olanı seç ve son cevabı 'Cevap:' ile ver."
    )
    return system, task["question"]


def meta_prompting(task: dict) -> tuple[str, str]:
    # Modele önce bu görev için iyi bir talimat yazdırıp sonra onu uygulatma.
    user = (
        "Önce bu görevi en iyi çözecek talimatı kendin tasarla, sonra o talimatı "
        f"uygulayarak çöz.\n\nGörev: {task['question']}"
    )
    return BASE_SYSTEM, user


@dataclass
class Strategy:
    """Bir prompt stratejisinin kimliği, görünen adı ve üretici fonksiyonu."""

    key: str
    name: str
    build: Callable[[dict], tuple[str, str]]


# Arayüz ve değerlendirme motorunun üzerinde döneceği strateji listesi.
STRATEGIES = [
    Strategy("zero_shot", "Zero-shot", zero_shot),
    Strategy("few_shot", "Few-shot", few_shot),
    Strategy("cot", "Chain-of-Thought", chain_of_thought),
    Strategy("react", "ReAct", react),
    Strategy("tot", "Tree-of-Thoughts", tree_of_thoughts),
    Strategy("meta", "Meta-prompting", meta_prompting),
]
