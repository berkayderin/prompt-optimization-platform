"""Prompt Optimizasyon Platformu - Streamlit arayuzu.

Bir gorev seti secilir, tum prompt stratejileri ayni set uzerinde calistirilir,
dogruluk/token/gecikme acisindan karsilastirilir ve en iyi strateji secilir.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.datasets import list_datasets, load_dataset
from core.evaluator import evaluate_strategy
from core.llm_client import LLMClient
from core.strategies import STRATEGIES

load_dotenv()

st.set_page_config(page_title="Prompt Optimizasyon Platformu", layout="wide")
st.title("Prompt Optimizasyon Platformu")
st.caption(
    "Farkli prompt stratejilerini ayni gorev setinde karsilastirir ve en iyisini secer."
)

provider = os.getenv("LLM_PROVIDER", "mock")

if provider == "mock":
    st.warning(
        "Su an MOCK modunda calisiyor; sonuclar temsilidir. Gercek metrikler icin "
        ".env dosyasinda bir saglayici (DeepSeek/GLM) ayarlayin."
    )

with st.sidebar:
    st.header("Ayarlar")
    st.write(f"Aktif saglayici: **{provider}**")
    dataset_name = st.selectbox("Gorev seti", list_datasets())
    temperature = st.slider("Sicaklik (temperature)", 0.0, 1.5, 0.7, 0.1)
    run = st.button("Stratejileri Calistir", type="primary")

if not run:
    st.info("Soldaki panelden bir gorev seti secip 'Stratejileri Calistir' butonuna basin.")
    st.stop()

data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

# Stratejileri sirayla calistirip ilerleme cubugunu guncelle.
progress = st.progress(0.0, "Calistiriliyor...")
results = []
for i, strategy in enumerate(STRATEGIES, start=1):
    results.append(evaluate_strategy(client, strategy, tasks, task_type, temperature))
    progress.progress(i / len(STRATEGIES), f"{strategy.name} tamamlandi")
progress.empty()

# Sonuclari tabloya cevir ve dogruluga gore sirala.
df = pd.DataFrame(
    [
        {
            "Strateji": r.strategy_name,
            "Dogruluk": round(r.accuracy, 3),
            "Ort. Token": round(r.avg_tokens, 1),
            "Ort. Gecikme (s)": round(r.avg_latency, 3),
        }
        for r in results
    ]
).sort_values("Dogruluk", ascending=False)

best = df.iloc[0]
st.success(f"En iyi strateji: **{best['Strateji']}** (dogruluk {best['Dogruluk']})")

col1, col2 = st.columns(2)
col1.subheader("Dogruluk")
col1.bar_chart(df.set_index("Strateji")["Dogruluk"])
col2.subheader("Ortalama Token (maliyet)")
col2.bar_chart(df.set_index("Strateji")["Ort. Token"])

st.subheader("Karsilastirma Tablosu")
st.dataframe(df, use_container_width=True)

with st.expander("Gorev bazinda ayrintilar"):
    for r in results:
        st.markdown(f"**{r.strategy_name}**")
        st.dataframe(pd.DataFrame(r.details), use_container_width=True)
