"""Prompt Optimizasyon Platformu - Streamlit arayüzü.

Bir görev seti seçilir, tüm prompt stratejileri aynı set üzerinde çalıştırılır,
doğruluk/token/gecikme açısından karşılaştırılır ve en iyi strateji seçilir.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.datasets import dataset_titles, list_datasets, load_dataset
from core.evaluator import evaluate_strategy
from core.llm_client import LLMClient
from core.strategies import STRATEGIES

load_dotenv()

st.set_page_config(page_title="Prompt Optimizasyon Platformu", layout="wide")

# Yardım metni: sağ üst köşedeki açılır pencerede gösterilir.
YARDIM_METNI = """
### Bu platform ne yapar?
Verdiğiniz görev setinde **6 farklı prompt stratejisini** (Zero-shot, Few-shot,
Chain-of-Thought, ReAct, Tree-of-Thoughts, Meta-prompting) aynı koşullarda
çalıştırır, **doğruluk** ve **maliyet (token)** açısından karşılaştırır ve
**en iyi stratejiyi** seçer.

### Nasıl kullanılır? (3 adım)
1. Soldaki panelden bir **görev seti** seçin (aritmetik veya duygu analizi).
2. İsterseniz **sıcaklık** değerini ayarlayın.
3. **"Stratejileri Çalıştır"** düğmesine basın; sonuçlar grafik ve tablo olarak gelir.

### Lokalde nasıl çalıştırılır?
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Gerçek sonuçlar için
Varsayılan **MOCK** modu yalnızca akışı gösterir (doğruluk 0 çıkar). Gerçek
metrikler için `.env` dosyasında bir sağlayıcı ayarlayın:
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
```
"""

# --- Başlık satırı: solda başlık, sağ üstte yardım düğmesi ---
sol, sag = st.columns([0.82, 0.18])
with sol:
    st.title("Prompt Optimizasyon Platformu")
    st.caption("Farklı prompt stratejilerini karşılaştırır ve en iyisini seçer.")
with sag:
    st.write("")  # dikey hizalama için boşluk
    with st.popover("Nasıl Çalışır?", use_container_width=True):
        st.markdown(YARDIM_METNI)

provider = os.getenv("LLM_PROVIDER", "mock")
if provider == "mock":
    st.warning(
        "Şu an MOCK modunda; sonuçlar temsilidir (doğruluk 0 çıkar). Gerçek "
        "metrikler için sağ üstteki \"Nasıl Çalışır?\" bölümüne bakın."
    )

# --- Sol panel: ayarlar ---
titles = dataset_titles()
with st.sidebar:
    st.header("Ayarlar")
    st.write(f"Aktif sağlayıcı: **{provider}**")
    st.divider()
    dataset_name = st.selectbox(
        "1) Görev seti",
        list_datasets(),
        format_func=lambda stem: titles.get(stem, stem),
        help="Stratejilerin üzerinde test edileceği görev kümesi.",
    )
    temperature = st.slider(
        "2) Sıcaklık (temperature)",
        0.0, 1.5, 0.7, 0.1,
        help="Düşük değer daha tutarlı, yüksek değer daha yaratıcı yanıt üretir.",
    )
    run = st.button("3) Stratejileri Çalıştır", type="primary", use_container_width=True)

# --- Çalıştırma ---
if not run:
    st.info("Başlamak için soldaki 1-2-3 adımlarını izleyin.")
    st.stop()

data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

progress = st.progress(0.0, "Çalıştırılıyor...")
results = []
for i, strategy in enumerate(STRATEGIES, start=1):
    results.append(evaluate_strategy(client, strategy, tasks, task_type, temperature))
    progress.progress(i / len(STRATEGIES), f"{strategy.name} tamamlandı")
progress.empty()

df = pd.DataFrame(
    [
        {
            "Strateji": r.strategy_name,
            "Doğruluk": round(r.accuracy, 3),
            "Ort. Token": round(r.avg_tokens, 1),
            "Ort. Gecikme (s)": round(r.avg_latency, 3),
        }
        for r in results
    ]
).sort_values("Doğruluk", ascending=False)

best = df.iloc[0]
st.success(f"En iyi strateji: {best['Strateji']} (doğruluk {best['Doğruluk']})")

# Özet ölçümler
m1, m2, m3 = st.columns(3)
m1.metric("Görev sayısı", len(tasks))
m2.metric("En yüksek doğruluk", f"{best['Doğruluk']:.0%}")
m3.metric("En iyinin token maliyeti", f"{best['Ort. Token']:.0f}")

st.divider()

col1, col2 = st.columns(2)
col1.subheader("Doğruluk")
col1.bar_chart(df.set_index("Strateji")["Doğruluk"])
col2.subheader("Ortalama Token (maliyet)")
col2.bar_chart(df.set_index("Strateji")["Ort. Token"])

st.subheader("Karşılaştırma Tablosu")
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Görev bazında ayrıntılar"):
    for r in results:
        st.markdown(f"**{r.strategy_name}**")
        st.dataframe(pd.DataFrame(r.details), use_container_width=True, hide_index=True)
