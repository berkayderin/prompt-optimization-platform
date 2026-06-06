"""Degerlendirme motoru: stratejileri gorev seti uzerinde calistirip skorlar.

Her gorev icin modelin yanitindan cevap cikarilir (answer extraction) ve beklenen
cevapla karsilastirilir. Acik uclu gorevlerde (ozetleme gibi) tek bir dogru cevap
olmadigindan puanlama, modelin hakem olarak kullanildigi LLM-as-judge yontemiyle
yapilir. Her gorev istenirse N kez calistirilarak yanitlarin kararliligi
(tutarlilik / consistency) da olculur. Sonuc olarak her strateji icin dogruluk,
tutarlilik, ortalama token ve ortalama gecikme raporlanir. Tum stratejiler AYNI
gorev seti uzerinde calistigi icin karsilastirma kontrolludur.
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
# Temsili tutarlilik: kisa, sablonlu yanit ureten stratejiler daha kararli;
# uzun muhakeme ureten stratejiler calistirmadan calistirmaya daha cok degisir.
_SIM_TUTARLILIK = {
    "zero_shot": 0.95,
    "few_shot": 0.92,
    "cot": 0.85,
    "react": 0.82,
    "tot": 0.80,
    "meta": 0.84,
}

# --- LLM-as-judge: acik uclu gorevlerin puanlanmasi ---
# Ozetleme gibi tek dogru cevabi olmayan gorevlerde exact-match calismaz;
# bunun yerine modele hakem rolu verilir ve yanitin referans cevapla ayni
# bilgiyi tasiyip tasimadigi sorulur (sicaklik 0 ile, kararli olsun diye).
JUDGE_SYSTEM = (
    "Sen titiz bir degerlendiricisin. Bir model yanitinin referans cevapla "
    "ozunde ayni bilgiyi tasiyip tasimadigina karar verirsin. "
    "Yalnizca EVET veya HAYIR yaz."
)


def _judge_correct(client: LLMClient, question: str, answer: str, expected: str) -> bool:
    """Yaniti, modeli hakem olarak kullanip referans cevapla karsilastirir."""
    user = (
        f"Gorev: {question}\n\n"
        f"Referans cevap: {expected}\n\n"
        f"Model yaniti: {answer}\n\n"
        "Model yaniti, referans cevapla ozunde ayni bilgiyi tasiyor mu? "
        "Yalnizca EVET veya HAYIR yaz."
    )
    resp = client.complete(JUDGE_SYSTEM, user, temperature=0.0)
    return "evet" in resp.text.strip().lower()


def _birim_hash(metin: str) -> float:
    """Metni 0 ile 1 arasinda kararli (deterministik) bir sayiya esler."""
    h = hashlib.sha1(metin.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _extract_number(text: str) -> str | None:
    """Metindeki son sayiyi cevap olarak kabul eder (aritmetik gorevler icin)."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def _is_correct(answer: str, expected: str, task_type: str) -> bool:
    """Model yanitinin beklenen cevapla eslesip eslesmedigini belirler (kural tabanli)."""
    answer = (answer or "").strip().lower()
    expected = str(expected).strip().lower()

    if task_type == "arithmetic":
        got, want = _extract_number(answer), _extract_number(expected)
        return got is not None and want is not None and float(got) == float(want)

    # Siniflandirma vb.: beklenen etiketin yanit metninde gecmesi yeterli.
    return expected in answer


def is_correct(client: LLMClient, text: str, task: dict, task_type: str) -> bool:
    """Gorev tipine gore dogru puanlama yontemini secer.

    Aritmetik ve siniflandirmada kural tabanli eslesme (exact match / etiket arama),
    ozetleme gibi acik uclu gorevlerde LLM-as-judge kullanilir.
    """
    if task_type == "summarization":
        return _judge_correct(client, task["question"], text, task["answer"])
    return _is_correct(text, task["answer"], task_type)


@dataclass
class StrategyResult:
    """Bir stratejinin gorev seti uzerindeki toplu sonucu."""

    strategy_key: str
    strategy_name: str
    accuracy: float
    avg_tokens: float
    avg_latency: float
    consistency: float = 1.0  # N=1 calistirmada tanim geregi 1.0
    details: list = field(default_factory=list)


def _simulate_strategy(
    strategy: Strategy, tasks: list[dict], task_type: str, runs: int = 1
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
    elif task_type == "summarization":
        # Acik uclu uretimde de fark, akil yurutme gorevlerine gore kucuktur.
        basari = 0.55 + (basari - 0.45) * 0.7

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

    # Tek calistirmada tutarlilik tanim geregi 1.0'dir; N>1'de stratejiye ozgu
    # temsili kararlilik degeri kullanilir.
    tutarlilik = 1.0 if runs <= 1 else _SIM_TUTARLILIK.get(strategy.key, 0.85)

    return StrategyResult(
        strategy_key=strategy.key,
        strategy_name=strategy.name,
        accuracy=k / n,
        avg_tokens=total_tokens / n,
        avg_latency=0.05,
        consistency=tutarlilik,
        details=details,
    )


def evaluate_strategy(
    client: LLMClient,
    strategy: Strategy,
    tasks: list[dict],
    task_type: str,
    temperature: float = 0.7,
    runs: int = 1,
) -> StrategyResult:
    """Tek bir stratejiyi tum gorevlerde calistirir ve metrikleri toplar.

    runs > 1 verilirse her soru N kez sorulur; dogruluk tum calistirmalarin
    ortalamasi, tutarlilik ise ayni sorunun calistirmalar arasi kararliligidir
    (gorev basina cogunluk orani: max(dogru, yanlis)/N, gorevler uzerinden
    ortalanir). Hakem (judge) cagrilarinin token/gecikme maliyeti, stratejinin
    kendi maliyetini olcmek icin metriklere DAHIL EDILMEZ.
    """
    # Demo (mock) modunda gercek cagri yerine temsili sonuc uretilir.
    if getattr(client, "provider", None) == "mock":
        return _simulate_strategy(strategy, tasks, task_type, runs)

    runs = max(1, runs)
    correct = 0
    total_tokens = 0
    total_latency = 0.0
    consistency_sum = 0.0
    details = []

    for task in tasks:
        system, user = strategy.build(task)
        dogru_sayisi = 0
        ilk_yanit = None
        for _ in range(runs):
            resp = client.complete(system, user, temperature)
            if ilk_yanit is None:
                ilk_yanit = resp.text
            dogru_sayisi += int(is_correct(client, resp.text, task, task_type))
            total_tokens += resp.total_tokens
            total_latency += resp.latency_s

        correct += dogru_sayisi
        consistency_sum += max(dogru_sayisi, runs - dogru_sayisi) / runs
        details.append(
            {
                "Soru": task["question"],
                "Beklenen": task["answer"],
                "Yanıt": ilk_yanit,
                "Doğru": f"{dogru_sayisi}/{runs}",
                "Token": total_tokens,
            }
        )

    n = len(tasks) or 1
    return StrategyResult(
        strategy_key=strategy.key,
        strategy_name=strategy.name,
        accuracy=correct / (n * runs),
        avg_tokens=total_tokens / (n * runs),
        avg_latency=total_latency / (n * runs),
        consistency=consistency_sum / n,
        details=details,
    )


def evaluate_all(
    client: LLMClient,
    strategies: list[Strategy],
    tasks: list[dict],
    task_type: str,
    temperature: float = 0.7,
    runs: int = 1,
) -> list[StrategyResult]:
    """Tum stratejileri ayni gorev seti uzerinde sirayla degerlendirir."""
    return [
        evaluate_strategy(client, s, tasks, task_type, temperature, runs)
        for s in strategies
    ]
