"""Degerlendirme motoru: stratejileri gorev seti uzerinde calistirip skorlar.

Her gorev icin modelin yanitindan cevap cikarilir (answer extraction) ve beklenen
cevapla karsilastirilir. Sonuc olarak her strateji icin dogruluk, ortalama token ve
ortalama gecikme raporlanir. Tum stratejiler AYNI gorev seti uzerinde calistigi icin
karsilastirma kontrolludur.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from core.llm_client import LLMClient
from core.strategies import Strategy

# --- Demo (mock) simulasyonu icin sabitler ---
# Anahtar yokken arayuzun anlamli gorunmesi icin her stratejiye temsili bir
# basari duzeyi ve ortalama uretilen token sayisi atanir. Bu degerler GERCEK
# olcum degildir; yalnizca demo amaclidir. Gercek sonuclar bir saglayici
# (DeepSeek/GLM) ayarlanarak elde edilir.
_SIM_BASARI = {
    "zero_shot": 0.45,
    "few_shot": 0.62,
    "cot": 0.82,
    "react": 0.70,
    "tot": 0.78,
    "meta": 0.72,
}
_SIM_URETILEN_TOKEN = {
    "zero_shot": 8,
    "few_shot": 10,
    "cot": 45,
    "react": 50,
    "tot": 60,
    "meta": 40,
}


def _birim_hash(metin: str) -> float:
    """Metni 0 ile 1 arasinda kararli (deterministik) bir sayiya esler."""
    h = hashlib.sha1(metin.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _extract_number(text: str) -> str | None:
    """Metindeki son sayiyi cevap olarak kabul eder (aritmetik gorevler icin)."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _is_correct(answer: str, expected: str, task_type: str) -> bool:
    """Model yanitinin beklenen cevapla eslesip eslesmedigini belirler."""
    answer = (answer or "").strip().lower()
    expected = str(expected).strip().lower()

    if task_type == "arithmetic":
        got, want = _extract_number(answer), _extract_number(expected)
        return got is not None and want is not None and float(got) == float(want)

    # Siniflandirma vb.: beklenen etiketin yanit metninde gecmesi yeterli.
    return expected in answer


@dataclass
class StrategyResult:
    """Bir stratejinin gorev seti uzerindeki toplu sonucu."""

    strategy_key: str
    strategy_name: str
    accuracy: float
    avg_tokens: float
    avg_latency: float
    details: list = field(default_factory=list)


def _simulate_strategy(
    strategy: Strategy, tasks: list[dict], task_type: str
) -> StrategyResult:
    """Demo modunda stratejiyi gercekci ve farkli sonuclarla taklit eder.

    Gercek model cagrisi yapmaz; her gorev icin dogru/yanlis karari, strateji ve
    soru metninden uretilen kararli bir hash ile verilir. Sonuclar [SIM] olarak
    isaretlenir ve temsilidir.
    """
    basari = _SIM_BASARI.get(strategy.key, 0.5)
    if task_type == "classification":
        # Siniflandirma gorevlerinde stratejiler arasi fark daha kucuktur.
        basari = 0.6 + (basari - 0.45) * 0.6

    # Dogruluk orani hedeflenen basari duzeyine yakin olsun diye: gorevler
    # stratejiye ozgu hash'e gore siralanir ve en dusuk hash'li k tanesi dogru
    # sayilir. Boylece kac gorevin dogru oldugu kararlidir, hangilerinin dogru
    # oldugu ise stratejiye gore degisir.
    n = len(tasks) or 1
    k = round(basari * len(tasks))
    sirali = sorted(
        range(len(tasks)),
        key=lambda i: _birim_hash(f"{strategy.key}:{tasks[i]['question']}"),
    )
    dogru_indexler = set(sirali[:k])

    total_tokens = 0
    details = []
    for i, task in enumerate(tasks):
        system, user = strategy.build(task)
        ok = i in dogru_indexler
        tokens = len((system + " " + user).split()) + _SIM_URETILEN_TOKEN.get(strategy.key, 10)

        total_tokens += tokens
        details.append(
            {
                "Soru": task["question"],
                "Beklenen": task["answer"],
                "Yanıt": f"[SIM] {task['answer'] if ok else 'yanlış/eksik yanıt'}",
                "Doğru": ok,
                "Token": tokens,
            }
        )

    return StrategyResult(
        strategy_key=strategy.key,
        strategy_name=strategy.name,
        accuracy=k / n,
        avg_tokens=total_tokens / n,
        avg_latency=0.05,
        details=details,
    )


def evaluate_strategy(
    client: LLMClient,
    strategy: Strategy,
    tasks: list[dict],
    task_type: str,
    temperature: float = 0.7,
) -> StrategyResult:
    """Tek bir stratejiyi tum gorevlerde calistirir ve metrikleri toplar."""
    # Demo (mock) modunda gercek cagri yerine temsili sonuc uretilir.
    if getattr(client, "provider", None) == "mock":
        return _simulate_strategy(strategy, tasks, task_type)

    correct = 0
    total_tokens = 0
    total_latency = 0.0
    details = []

    for task in tasks:
        system, user = strategy.build(task)
        resp = client.complete(system, user, temperature)
        ok = _is_correct(resp.text, task["answer"], task_type)

        correct += int(ok)
        total_tokens += resp.total_tokens
        total_latency += resp.latency_s
        details.append(
            {
                "Soru": task["question"],
                "Beklenen": task["answer"],
                "Yanıt": resp.text,
                "Doğru": ok,
                "Token": resp.total_tokens,
            }
        )

    n = len(tasks) or 1
    return StrategyResult(
        strategy_key=strategy.key,
        strategy_name=strategy.name,
        accuracy=correct / n,
        avg_tokens=total_tokens / n,
        avg_latency=total_latency / n,
        details=details,
    )


def evaluate_all(
    client: LLMClient,
    strategies: list[Strategy],
    tasks: list[dict],
    task_type: str,
    temperature: float = 0.7,
) -> list[StrategyResult]:
    """Tum stratejileri ayni gorev seti uzerinde sirayla degerlendirir."""
    return [
        evaluate_strategy(client, s, tasks, task_type, temperature) for s in strategies
    ]
