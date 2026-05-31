"""Meta-prompting ile otomatik prompt iyilestirme.

Bir baslangic talimatini, modele "bu talimati daha etkili hale getir" dedirterek
tur tur iyilestirir ve her turda gorev seti uzerinde test eder. Yeni talimat skoru
artirirsa benimsenir; aksi halde mevcut en iyi korunur.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.evaluator import _is_correct
from core.llm_client import LLMClient


@dataclass
class OptimizationStep:
    """Optimizasyon dongusunde tek bir turun talimati ve skoru."""

    iteration: int
    instruction: str
    accuracy: float


def _score(
    client: LLMClient,
    instruction: str,
    tasks: list[dict],
    task_type: str,
    temperature: float,
) -> float:
    """Verilen sistem talimatinin gorev setindeki dogrulugunu olcer."""
    correct = 0
    for task in tasks:
        resp = client.complete(instruction, task["question"], temperature)
        correct += int(_is_correct(resp.text, task["answer"], task_type))
    return correct / (len(tasks) or 1)


def _simulate_optimization(
    seed_instruction: str, rounds: int
) -> tuple[str, float, list[OptimizationStep]]:
    """Demo modunda tur tur artan temsili bir iyilestirme egrisi uretir."""
    history = [OptimizationStep(0, seed_instruction, 0.50)]
    score = 0.50
    for i in range(1, rounds + 1):
        score = min(0.90, score + 0.12)  # her turda kararli bir artis
        history.append(
            OptimizationStep(i, f"[SIM] Tur {i} sonunda iyilestirilmis talimat", round(score, 3))
        )
    best = max(history, key=lambda s: s.accuracy)
    return best.instruction, best.accuracy, history


def optimize(
    client: LLMClient,
    seed_instruction: str,
    tasks: list[dict],
    task_type: str,
    rounds: int = 3,
    temperature: float = 0.7,
) -> tuple[str, float, list[OptimizationStep]]:
    """Talimati tur tur iyilestirir; en iyi talimati, skorunu ve gecmisi dondurur."""
    # Demo (mock) modunda gercek cagri yerine temsili bir egri uretilir.
    if getattr(client, "provider", None) == "mock":
        return _simulate_optimization(seed_instruction, rounds)

    best_instruction = seed_instruction
    best_score = _score(client, best_instruction, tasks, task_type, temperature)
    history = [OptimizationStep(0, best_instruction, best_score)]

    for i in range(1, rounds + 1):
        # Meta-prompt: modelden mevcut talimatin daha iyi bir surumunu iste.
        meta = (
            "Asagidaki sistem talimatini, gorevdeki dogrulugu artiracak sekilde "
            "yeniden yaz. Yalnizca yeni talimati dondur, aciklama ekleme.\n\n"
            f"Mevcut talimat:\n{best_instruction}"
        )
        proposal = client.complete(
            "Sen deneyimli bir prompt muhendisisin.", meta, temperature
        ).text.strip()

        score = _score(client, proposal, tasks, task_type, temperature)
        history.append(OptimizationStep(i, proposal, score))

        if score > best_score:
            best_instruction, best_score = proposal, score

    return best_instruction, best_score, history
