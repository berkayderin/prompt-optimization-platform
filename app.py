"""Prompt Optimizasyon Platformu - Streamlit arayüzü.

Üç bölümden oluşur:
  1. Karşılaştırma : tüm stratejileri aynı görev setinde kıyaslar.
  2. Optimizasyon  : meta-prompting ile bir talimatı tur tur iyileştirir.
  3. A/B Testi     : iki stratejiyi doğrudan karşılaştırır.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.datasets import dataset_titles, list_datasets, load_dataset
from core.evaluator import evaluate_strategy
from core.llm_client import LLMClient
from core.optimizer import optimize
from core.strategies import BASE_SYSTEM, STRATEGIES
from core.versioning import PromptVersion, compare, prompt_id

load_dotenv()

st.set_page_config(page_title="Prompt Optimizasyon Platformu", layout="centered")

YARDIM_METNI = """
**Ne yapar?** Verdiğiniz görevde farklı prompt stratejilerini dener,
doğruluk ve maliyet açısından karşılaştırır, en iyisini seçer.

**Nasıl kullanılır?**
1. Soldan bir görev seti seçin.
2. Bir sekme açıp "Çalıştır" düğmesine basın.

**Gerçek sonuçlar için** `.env` dosyasında bir sağlayıcı ayarlayın:
`LLM_PROVIDER=deepseek` ve `DEEPSEEK_API_KEY=...`
"""

# --- Başlık ve yardım ---
sol, sag = st.columns([0.75, 0.25])
sol.title("Prompt Optimizasyon Platformu")
with sag:
    st.write("")
    with st.popover("Yardım", use_container_width=True):
        st.markdown(YARDIM_METNI)

provider = os.getenv("LLM_PROVIDER", "mock")
if provider == "mock":
    st.info("MOCK modunda çalışıyor; gerçek sonuçlar için bir sağlayıcı ayarlayın (Yardım).")

# --- Sol panel: ortak ayarlar ---
titles = dataset_titles()
with st.sidebar:
    st.header("Ayarlar")
    dataset_name = st.selectbox(
        "Görev seti",
        list_datasets(),
        format_func=lambda stem: titles.get(stem, stem),
    )
    temperature = st.slider("Sıcaklık", 0.0, 1.5, 0.7, 0.1)

data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

sekme1, sekme2, sekme3 = st.tabs(["Karşılaştırma", "Optimizasyon", "A/B Testi"])


# --- Sekme 1: tüm stratejileri karşılaştır ---
with sekme1:
    st.write("Tüm stratejileri aynı görev setinde karşılaştırır.")
    if st.button("Çalıştır", type="primary", key="btn_karsilastir"):
        with st.spinner("Çalıştırılıyor..."):
            results = [
                evaluate_strategy(client, s, tasks, task_type, temperature)
                for s in STRATEGIES
            ]

        df = pd.DataFrame(
            [
                {
                    "Strateji": r.strategy_name,
                    "Doğruluk": round(r.accuracy, 3),
                    "Token": round(r.avg_tokens, 1),
                }
                for r in results
            ]
        ).sort_values("Doğruluk", ascending=False)

        st.success(f"En iyi strateji: {df.iloc[0]['Strateji']}")
        st.bar_chart(df.set_index("Strateji")["Doğruluk"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Sonuçları indir (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name=f"sonuclar_{dataset_name}.csv",
            mime="text/csv",
        )


# --- Sekme 2: meta-prompting ile otomatik iyileştirme ---
with sekme2:
    st.write("Bir talimatı tur tur iyileştirir ve en iyisini bulur.")
    seed = st.text_area("Başlangıç talimatı", value=BASE_SYSTEM, height=70)
    rounds = st.slider("Tur sayısı", 1, 5, 3, key="opt_rounds")

    if st.button("Çalıştır", type="primary", key="btn_optimize"):
        with st.spinner("İyileştiriliyor..."):
            best_instruction, best_score, history = optimize(
                client, seed, tasks, task_type, rounds, temperature
            )

        st.success(f"En iyi doğruluk: {best_score:.0%}")
        st.code(best_instruction)

        hist_df = pd.DataFrame(
            [{"Tur": s.iteration, "Doğruluk": round(s.accuracy, 3)} for s in history]
        )
        st.line_chart(hist_df.set_index("Tur")["Doğruluk"])


# --- Sekme 3: iki stratejiyi karşılaştır (A/B) ---
with sekme3:
    st.write("Seçtiğiniz iki stratejiyi karşılaştırır.")
    isimler = [s.name for s in STRATEGIES]
    c1, c2 = st.columns(2)
    a_ad = c1.selectbox("A", isimler, index=0, key="ab_a")
    b_ad = c2.selectbox("B", isimler, index=2, key="ab_b")

    if st.button("Çalıştır", type="primary", key="btn_ab"):
        strat_a = next(s for s in STRATEGIES if s.name == a_ad)
        strat_b = next(s for s in STRATEGIES if s.name == b_ad)

        with st.spinner("Çalıştırılıyor..."):
            ra = evaluate_strategy(client, strat_a, tasks, task_type, temperature)
            rb = evaluate_strategy(client, strat_b, tasks, task_type, temperature)

        va = PromptVersion(prompt_id(strat_a.key), strat_a.key, strat_a.name, ra.accuracy, ra.avg_tokens)
        vb = PromptVersion(prompt_id(strat_b.key), strat_b.key, strat_b.name, rb.accuracy, rb.avg_tokens)
        kazanan = a_ad if compare(va, vb)["kazanan"] == va.pid else b_ad

        st.success(f"Kazanan: {kazanan}")
        c1, c2 = st.columns(2)
        c1.metric(a_ad, f"{ra.accuracy:.0%}", f"{ra.avg_tokens:.0f} token")
        c2.metric(b_ad, f"{rb.accuracy:.0%}", f"{rb.avg_tokens:.0f} token")
