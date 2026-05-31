"""Degerlendirme motoru: stratejileri gorev seti uzerinde calistirip skorlar.

Her gorev icin modelin yanitindan cevap cikarilir (answer extraction) ve beklenen
cevapla karsilastirilir. Sonuc olarak her strateji icin dogruluk, ortalama token ve
ortalama gecikme raporlanir. Tum stratejiler AYNI gorev seti uzerinde calistigi icin
karsilastirma kontrolludur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.llm_client import LLMClient
from core.strategies import Strategy


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


def evaluate_strategy(
    client: LLMClient,
    strategy: Strategy,
    tasks: list[dict],
    task_type: str,
    temperature: float = 0.7,
) -> StrategyResult:
    """Tek bir stratejiyi tum gorevlerde calistirir ve metrikleri toplar."""
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
