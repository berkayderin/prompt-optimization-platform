"""Prompt Optimizasyon Platformu - Streamlit arayüzü.

Üç bölümden oluşur:
  1. Strateji Karşılaştırma : tüm stratejileri aynı görev setinde kıyaslar.
  2. Otomatik Optimizasyon  : meta-prompting ile bir talimatı tur tur iyileştirir.
  3. A/B Testi              : iki stratejiyi doğrudan karşılaştırır.
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

st.set_page_config(page_title="Prompt Optimizasyon Platformu", layout="wide")

YARDIM_METNI = """
### Bu platform ne yapar?
Verdiğiniz görev setinde **6 farklı prompt stratejisini** (Zero-shot, Few-shot,
Chain-of-Thought, ReAct, Tree-of-Thoughts, Meta-prompting) aynı koşullarda
çalıştırır, **doğruluk** ve **maliyet (token)** açısından karşılaştırır ve
**en iyi stratejiyi** seçer.

### Üç bölüm
- **Strateji Karşılaştırma:** Tüm stratejileri aynı görev setinde kıyaslar.
- **Otomatik Optimizasyon:** Bir başlangıç talimatını tur tur iyileştirir.
- **A/B Testi:** Seçtiğiniz iki stratejiyi doğrudan karşılaştırır.

### Nasıl kullanılır?
1. Soldaki panelden bir **görev seti** ve **sıcaklık** seçin.
2. Üstteki sekmelerden birini açıp ilgili **çalıştır** düğmesine basın.

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

# --- Sol panel: tüm bölümlerce paylaşılan ayarlar ---
titles = dataset_titles()
with st.sidebar:
    st.header("Ayarlar")
    st.write(f"Aktif sağlayıcı: **{provider}**")
    st.divider()
    dataset_name = st.selectbox(
        "Görev seti",
        list_datasets(),
        format_func=lambda stem: titles.get(stem, stem),
        help="Stratejilerin üzerinde test edileceği görev kümesi.",
    )
    temperature = st.slider(
        "Sıcaklık (temperature)",
        0.0, 1.5, 0.7, 0.1,
        help="Düşük değer daha tutarlı, yüksek değer daha yaratıcı yanıt üretir.",
    )

# Seçilen görev seti tüm sekmelerde kullanılır.
data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

sekme1, sekme2, sekme3 = st.tabs(
    ["Strateji Karşılaştırma", "Otomatik Optimizasyon", "A/B Testi"]
)


# --- Sekme 1: tüm stratejileri karşılaştır ---
with sekme1:
    st.subheader("Tüm stratejileri aynı görev setinde karşılaştır")
    if st.button("Stratejileri Çalıştır", type="primary", key="btn_karsilastir"):
        progress = st.progress(0.0, "Çalıştırılıyor...")
        results = []
        for i, strategy in enumerate(STRATEGIES, start=1):
            results.append(
                evaluate_strategy(client, strategy, tasks, task_type, temperature)
            )
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
                st.dataframe(
                    pd.DataFrame(r.details), use_container_width=True, hide_index=True
                )
    else:
        st.info("Çalıştırmak için yukarıdaki düğmeye basın.")


# --- Sekme 2: meta-prompting ile otomatik iyileştirme ---
with sekme2:
    st.subheader("Bir talimatı tur tur iyileştir")
    st.caption(
        "Model, mevcut talimatı her turda yeniden yazar ve görev setinde test eder; "
        "skoru artıran talimat benimsenir."
    )
    seed = st.text_area("Başlangıç talimatı", value=BASE_SYSTEM, height=80)
    rounds = st.slider("Tur sayısı", 1, 5, 3, key="opt_rounds")

    if st.button("Optimizasyonu Başlat", type="primary", key="btn_optimize"):
        with st.spinner("İyileştiriliyor..."):
            best_instruction, best_score, history = optimize(
                client, seed, tasks, task_type, rounds, temperature
            )

        st.success(f"En iyi doğruluk: {best_score:.0%}")
        st.markdown("**Bulunan en iyi talimat:**")
        st.code(best_instruction)

        hist_df = pd.DataFrame(
            [
                {"Tur": s.iteration, "Doğruluk": round(s.accuracy, 3), "Talimat": s.instruction}
                for s in history
            ]
        )
        st.subheader("Tur tur doğruluk")
        st.line_chart(hist_df.set_index("Tur")["Doğruluk"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.info("Başlatmak için talimatı düzenleyip düğmeye basın.")


# --- Sekme 3: iki stratejiyi karşılaştır (A/B) ---
with sekme3:
    st.subheader("İki stratejiyi doğrudan karşılaştır")
    isimler = [s.name for s in STRATEGIES]
    c1, c2 = st.columns(2)
    a_ad = c1.selectbox("A stratejisi", isimler, index=0, key="ab_a")
    b_ad = c2.selectbox("B stratejisi", isimler, index=2, key="ab_b")

    if st.button("A/B Testini Çalıştır", type="primary", key="btn_ab"):
        strat_a = next(s for s in STRATEGIES if s.name == a_ad)
        strat_b = next(s for s in STRATEGIES if s.name == b_ad)

        with st.spinner("Çalıştırılıyor..."):
            ra = evaluate_strategy(client, strat_a, tasks, task_type, temperature)
            rb = evaluate_strategy(client, strat_b, tasks, task_type, temperature)

        va = PromptVersion(prompt_id(strat_a.key), strat_a.key, strat_a.name, ra.accuracy, ra.avg_tokens)
        vb = PromptVersion(prompt_id(strat_b.key), strat_b.key, strat_b.name, rb.accuracy, rb.avg_tokens)
        sonuc = compare(va, vb)
        kazanan = a_ad if sonuc["kazanan"] == va.pid else b_ad

        st.success(f"Kazanan: {kazanan}")
        c1, c2 = st.columns(2)
        c1.metric(a_ad, f"{ra.accuracy:.0%}", f"{ra.avg_tokens:.0f} token")
        c2.metric(b_ad, f"{rb.accuracy:.0%}", f"{rb.avg_tokens:.0f} token")

        st.write(
            f"Doğruluk farkı (A-B): **{sonuc['dogruluk_farki']}** · "
            f"Token farkı (A-B): **{sonuc['token_farki']}**"
        )
    else:
        st.info("İki strateji seçip düğmeye basın.")
