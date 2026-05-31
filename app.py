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

st.set_page_config(page_title="Prompt Optimizasyon Platformu", page_icon="🧪", layout="wide")

# Yardim metni: sag ust kosedeki tooltip/popover icinde gosterilir.
YARDIM_METNI = """
### Bu platform ne yapar?
Verdiginiz gorev setinde **6 farkli prompt stratejisini** (Zero-shot, Few-shot,
Chain-of-Thought, ReAct, Tree-of-Thoughts, Meta-prompting) ayni kosullarda
calistirir, **dogruluk** ve **maliyet (token)** acisindan karsilastirir ve
**en iyi stratejiyi** secer.

### Nasil kullanilir? (3 adim)
1. Soldaki panelden bir **gorev seti** secin (aritmetik veya duygu analizi).
2. Isterseniz **sicaklik** degerini ayarlayin.
3. **"Stratejileri Calistir"** butonuna basin; sonuclar grafik ve tablo olarak gelir.

### Lokalde nasil calistirilir?
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Gercek sonuclar icin
Varsayilan **MOCK** modu yalnizca akisi gosterir (dogruluk 0 cikar). Gercek
metrikler icin `.env` dosyasinda bir saglayici ayarlayin:
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
```
"""

# --- Baslik satiri: solda baslik, sagda yardim butonu (tooltip) ---
sol, sag = st.columns([0.82, 0.18])
with sol:
    st.title("🧪 Prompt Optimizasyon Platformu")
    st.caption("Farkli prompt stratejilerini karsilastirir ve en iyisini secer.")
with sag:
    st.write("")  # dikey hizalama icin bosluk
    with st.popover("ℹ️ Nasil Calisir?", use_container_width=True):
        st.markdown(YARDIM_METNI)

provider = os.getenv("LLM_PROVIDER", "mock")
if provider == "mock":
    st.warning(
        "Su an **MOCK** modunda; sonuclar temsilidir (dogruluk 0 cikar). Gercek "
        "metrikler icin sag ustteki **Nasil Calisir?** bolumune bakin.",
        icon="⚠️",
    )

# --- Sol panel: ayarlar ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.write(f"Aktif saglayici: **{provider}**")
    st.divider()
    dataset_name = st.selectbox(
        "1) Gorev seti",
        list_datasets(),
        help="Stratejilerin uzerinde test edilecegi gorev kumesi.",
    )
    temperature = st.slider(
        "2) Sicaklik (temperature)",
        0.0, 1.5, 0.7, 0.1,
        help="Dusuk deger daha tutarli, yuksek deger daha yaratici yanit uretir.",
    )
    run = st.button("3) Stratejileri Calistir ▶", type="primary", use_container_width=True)

# --- Calistirma ---
if not run:
    st.info("Baslamak icin soldaki **1-2-3** adimlarini izleyin.", icon="👈")
    st.stop()

data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

progress = st.progress(0.0, "Calistiriliyor...")
results = []
for i, strategy in enumerate(STRATEGIES, start=1):
    results.append(evaluate_strategy(client, strategy, tasks, task_type, temperature))
    progress.progress(i / len(STRATEGIES), f"{strategy.name} tamamlandi")
progress.empty()

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
st.success(f"🏆 En iyi strateji: **{best['Strateji']}** (dogruluk {best['Dogruluk']})")

# Ozet metrikler
m1, m2, m3 = st.columns(3)
m1.metric("Gorev sayisi", len(tasks))
m2.metric("En yuksek dogruluk", f"{best['Dogruluk']:.0%}")
m3.metric("En iyinin token maliyeti", f"{best['Ort. Token']:.0f}")

st.divider()

col1, col2 = st.columns(2)
col1.subheader("Dogruluk")
col1.bar_chart(df.set_index("Strateji")["Dogruluk"])
col2.subheader("Ortalama Token (maliyet)")
col2.bar_chart(df.set_index("Strateji")["Ort. Token"])

st.subheader("Karsilastirma Tablosu")
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("🔍 Gorev bazinda ayrintilar"):
    for r in results:
        st.markdown(f"**{r.strategy_name}**")
        st.dataframe(pd.DataFrame(r.details), use_container_width=True, hide_index=True)
