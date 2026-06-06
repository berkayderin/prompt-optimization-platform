"""Tum gorev setlerinde tum stratejileri kosar; sonuclari data/results altina yazar.

Makaledeki deney tablolari ve grafikler bu ciktilardan beslenir. Kullanim:

    python run_experiments.py                      # .env'deki LLM_PROVIDER ile
    python run_experiments.py --provider deepseek  # saglayiciyi elle sec
    python run_experiments.py --runs 3             # tutarlilik icin her soruyu 3 kez sor

Cikti: gorev seti basina bir CSV (deney_<set>_<tarih>.csv) ve tum deneyin
tek bir JSON ozeti (deney_ozeti_<tarih>.json).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from core.datasets import list_datasets, load_dataset
from core.evaluator import evaluate_all
from core.llm_client import LLMClient
from core.strategies import STRATEGIES
from core.versioning import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tum gorev setlerinde tum stratejileri kosar ve sonuclari kaydeder."
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="mock | deepseek | glm | openai | ollama (varsayilan: .env'deki LLM_PROVIDER)",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Ornekleme sicakligi")
    parser.add_argument(
        "--runs", type=int, default=1, help="Tutarlilik olcumu icin soru basina tekrar sayisi"
    )
    args = parser.parse_args()

    load_dotenv()
    client = LLMClient(args.provider)
    if client.provider == "mock":
        print(
            "UYARI: mock (deneme) modundasiniz; sonuclar temsilidir, bilimsel olcum "
            "degildir. Gercek deney icin .env'de bir saglayici ayarlayin."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ozet = {
        "saglayici": client.provider,
        "model": client.model,
        "sicaklik": args.temperature,
        "tekrar_sayisi": args.runs,
        "tarih": stamp,
        "gorev_setleri": [],
    }

    for name in list_datasets():
        data = load_dataset(name)
        print(f"\n=== {data['name']} ({len(data['tasks'])} soru, tip: {data['task_type']}) ===")

        results = evaluate_all(
            client, STRATEGIES, data["tasks"], data["task_type"], args.temperature, args.runs
        )
        df = pd.DataFrame(
            [
                {
                    "Strateji": r.strategy_name,
                    "Doğruluk": round(r.accuracy, 3),
                    "Tutarlılık": round(r.consistency, 3),
                    "Token": round(r.avg_tokens, 1),
                    "Gecikme (s)": round(r.avg_latency, 2),
                }
                for r in results
            ]
        ).sort_values("Doğruluk", ascending=False)
        print(df.to_string(index=False))

        csv_yolu = RESULTS_DIR / f"deney_{name}_{stamp}.csv"
        df.to_csv(csv_yolu, index=False)
        ozet["gorev_setleri"].append(
            {
                "ad": data["name"],
                "tip": data["task_type"],
                "soru_sayisi": len(data["tasks"]),
                "dosya": csv_yolu.name,
                "sonuclar": df.to_dict(orient="records"),
            }
        )

    ozet_yolu = RESULTS_DIR / f"deney_ozeti_{stamp}.json"
    ozet_yolu.write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTum sonuclar kaydedildi: {RESULTS_DIR}")
    print(f"Ozet dosyasi: {ozet_yolu.name}")


if __name__ == "__main__":
    main()
