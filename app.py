"""Prompt Optimizasyon Platformu - Streamlit arayüzü.

Üç bölümden oluşur:
  1. Karşılaştırma : tüm stratejileri aynı görev setinde kıyaslar.
  2. Optimizasyon  : meta-prompting ile bir talimatı tur tur iyileştirir.
  3. A/B Testi     : iki stratejiyi doğrudan karşılaştırır.

Arayüz, bu konuyu bilmeyen kullanıcılar için adım adım yönlendirme içerir.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.datasets import dataset_titles, list_datasets, load_dataset
from core.evaluator import evaluate_all, evaluate_strategy
from core.llm_client import LLMClient
from core.optimizer import optimize
from core.strategies import BASE_SYSTEM, STRATEGIES, STRATEGY_REASONS
from core.versioning import PromptVersion, compare, prompt_id, save_version

load_dotenv()


def sonuc_aciklamasi(results: list, task_type: str) -> str:
    """Karşılaştırma sonucunu, kazananın nedenini ve maliyet dengesini sade
    dille açıklayan bir metin üretir."""
    sirali = sorted(results, key=lambda r: r.accuracy, reverse=True)
    en_iyi = sirali[0]
    cumleler = [
        f"En yüksek doğruluğu **{en_iyi.strategy_name}** verdi; çünkü bu yöntem "
        f"{STRATEGY_REASONS.get(en_iyi.strategy_key, '')}."
    ]

    en_basit = next((r for r in results if r.strategy_key == "zero_shot"), None)
    if en_basit and en_basit.strategy_key != en_iyi.strategy_key:
        fark = (en_iyi.accuracy - en_basit.accuracy) * 100
        if fark > 0:
            cumleler.append(
                f"En basit yöntem olan Zero-shot'a göre doğruluk yaklaşık "
                f"{fark:.0f} puan daha yüksek."
            )

    en_ucuz = min(results, key=lambda r: r.avg_tokens)
    if en_iyi.avg_tokens > en_ucuz.avg_tokens * 1.3:
        cumleler.append(
            f"Ancak bu doğruluğun bir bedeli var: {en_iyi.strategy_name} ortalama "
            f"{en_iyi.avg_tokens:.0f} token harcarken en ekonomik yöntem "
            f"({en_ucuz.strategy_name}) {en_ucuz.avg_tokens:.0f} token kullanıyor. "
            f"Yani daha yüksek doğruluk, daha yüksek maliyetle geliyor."
        )

    if task_type == "classification":
        cumleler.append(
            "Sınıflandırma görevlerinde yöntemler arası fark, akıl yürütme "
            "görevlerine göre genellikle daha küçüktür."
        )
    return " ".join(cumleler)

st.set_page_config(page_title="Prompt Optimizasyon Platformu", layout="centered")

# --- Başlık ve kısa, sade açıklama ---
st.title("Prompt Optimizasyon Platformu")
st.write(
    "Bu araç, bir yapay zeka modeline soru sormanın farklı yollarını (stratejileri) "
    "dener ve hangisinin en doğru sonucu verdiğini gösterir."
)

# --- Her zaman görünen adım adım kullanım kılavuzu ---
st.info(
    "Nasıl kullanılır?\n\n"
    "1. Soldaki panelden bir **görev seti** seçin.\n"
    "2. Aşağıdaki sekmelerden birini açın (başlangıç için **Karşılaştırma** önerilir).\n"
    "3. **Çalıştır** düğmesine basın ve sonuçları görün."
)

provider = os.getenv("LLM_PROVIDER", "mock")
if provider == "mock":
    st.caption(
        "Not: Şu an deneme modunda çalışıyor; sonuçlar örnek (simülasyon) verisidir, "
        "platformun nasıl çalıştığını göstermek içindir. Gerçek sonuçlar için bir "
        "yapay zeka sağlayıcısı (DeepSeek/GLM) ayarlanmalıdır."
    )

# --- Sol panel: adım adım ayarlar ---
titles = dataset_titles()
with st.sidebar:
    st.header("Ayarlar")
    st.markdown("**Adım 1 — Görev seti seçin**")
    dataset_name = st.selectbox(
        "Görev seti",
        list_datasets(),
        format_func=lambda stem: titles.get(stem, stem),
        help="Modele sorulacak hazır soruların bulunduğu küme.",
        label_visibility="collapsed",
    )
    st.markdown("**Adım 2 — Sıcaklık (isteğe bağlı)**")
    temperature = st.slider(
        "Sıcaklık",
        0.0, 1.5, 0.7, 0.1,
        help="Düşük değer daha tutarlı, yüksek değer daha yaratıcı yanıt verir. "
        "Emin değilseniz olduğu gibi bırakın.",
        label_visibility="collapsed",
    )

data = load_dataset(dataset_name)
tasks = data["tasks"]
task_type = data["task_type"]
client = LLMClient(provider)

# Seçili görev setinin ne olduğunu ve neden seçildiğini sade dille açıkla;
# konuya yabancı kullanıcı, hangi soru kümesiyle çalıştığını bilsin.
aciklama = data.get("description")
if aciklama:
    st.markdown(f"**Seçili görev seti: {data['name']}** ({len(tasks)} soru)")
    st.caption(aciklama)

with st.expander("Görev setleri neye göre seçildi?"):
    st.write(
        "Setler, farklı strateji türlerinin güçlü ve zayıf yanlarını ortaya "
        "çıkaracak şekilde seçildi:\n\n"
        "- **Aritmetik Problemler**: Adım adım düşünmeyi gerektiren sorular. "
        "Burada Chain-of-Thought gibi yöntemler öne çıkar.\n"
        "- **Duygu Analizi** ve **Konu Sınıflandırma**: Tek adımlı etiketleme "
        "görevleri. Burada yöntemler arası fark küçülür.\n\n"
        "Amaç şunu göstermektir: en iyi prompt stratejisi sabit değildir, "
        "çözülecek görevin türüne göre değişir."
    )

st.markdown("**Adım 3 — Bir sekme seçip çalıştırın**")
sekme1, sekme2, sekme3 = st.tabs(["Karşılaştırma", "Optimizasyon", "A/B Testi"])


# --- Sekme 1: tüm stratejileri karşılaştır ---
with sekme1:
    st.write(
        "Tüm soru sorma yöntemlerini dener ve en iyi sonucu vereni bulur. "
        "Başlamak için en uygun seçenek budur."
    )
    if st.button("Çalıştır", type="primary", key="btn_karsilastir"):
        with st.spinner("Çalıştırılıyor..."):
            results = evaluate_all(client, STRATEGIES, tasks, task_type, temperature)

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

        st.markdown("**Sonuç neden böyle?**")
        st.write(sonuc_aciklamasi(results, task_type))

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
    st.write(
        "İleri düzey: Yazdığınız bir talimatı, modelin kendisine tur tur "
        "iyileştirterek en iyi halini bulur. Karşılaştırma sekmesi hazır yöntemleri "
        "dener; bu sekme ise tek bir talimatı otomatik olarak geliştirir."
    )
    st.caption(
        "Nasıl çalışır: Başlangıç talimatı görev setinde test edilir. Sonra model, "
        "talimatı daha iyi hale getirmesi için yönlendirilir; yeni talimat tekrar "
        "test edilir. Skoru artıran talimat benimsenir. Bu, seçtiğiniz tur sayısı "
        "kadar tekrarlanır."
    )
    seed = st.text_area("Başlangıç talimatı", value=BASE_SYSTEM, height=70)
    rounds = st.slider("Tur sayısı", 1, 5, 3, key="opt_rounds")

    if st.button("Çalıştır", type="primary", key="btn_optimize"):
        with st.spinner("İyileştiriliyor..."):
            best_instruction, best_score, history = optimize(
                client, seed, tasks, task_type, rounds, temperature
            )

        st.success(f"En iyi doğruluk: {best_score:.0%}")

        ilk = history[0].accuracy
        kazanc = (best_score - ilk) * 100
        st.markdown("**Sonuç neden böyle?**")
        if kazanc > 0:
            st.write(
                f"Başlangıç talimatı %{ilk * 100:.0f} doğrulukla başladı; otomatik "
                f"iyileştirme sonunda %{best_score * 100:.0f} doğruluğa ulaştı "
                f"(yaklaşık {kazanc:.0f} puan artış). Aşağıdaki talimat, görev "
                "setinde en yüksek skoru veren sürümdür."
            )
        else:
            st.write(
                "Bu görev setinde iyileştirme turları başlangıç talimatını geçemedi; "
                "yani başlangıç talimatı zaten yeterince iyiydi. En iyi sürüm "
                "aşağıdadır."
            )
        st.code(best_instruction)

        hist_df = pd.DataFrame(
            [{"Tur": s.iteration, "Doğruluk": round(s.accuracy, 3)} for s in history]
        )
        st.line_chart(hist_df.set_index("Tur")["Doğruluk"])


# --- Sekme 3: iki stratejiyi karşılaştır (A/B) ---
with sekme3:
    st.write(
        "İleri düzey: Seçtiğiniz iki yöntemi birebir karşılaştırır. Tüm yöntemleri "
        "değil de yalnızca merak ettiğiniz iki tanesini kıyaslamak için kullanılır. "
        "Sonuç, her iki yöntemin doğruluğunu ve token maliyetini yan yana gösterir; "
        "ayrıca sonuçlar sürüm geçmişine kaydedilir."
    )
    isimler = [s.name for s in STRATEGIES]
    c1, c2 = st.columns(2)
    a_ad = c1.selectbox("Birinci yöntem", isimler, index=0, key="ab_a")
    b_ad = c2.selectbox("İkinci yöntem", isimler, index=2, key="ab_b")

    if st.button("Çalıştır", type="primary", key="btn_ab"):
        strat_a = next(s for s in STRATEGIES if s.name == a_ad)
        strat_b = next(s for s in STRATEGIES if s.name == b_ad)

        with st.spinner("Çalıştırılıyor..."):
            ra = evaluate_strategy(client, strat_a, tasks, task_type, temperature)
            rb = evaluate_strategy(client, strat_b, tasks, task_type, temperature)

        va = PromptVersion(prompt_id(strat_a.key), strat_a.key, strat_a.name, ra.accuracy, ra.avg_tokens)
        vb = PromptVersion(prompt_id(strat_b.key), strat_b.key, strat_b.name, rb.accuracy, rb.avg_tokens)

        # Karşılaştırılan iki sürümü, sonradan izlenebilmesi için sürüm
        # geçmişine kaydet (data/results/versions.json).
        save_version(va)
        save_version(vb)

        kazanan = a_ad if compare(va, vb)["kazanan"] == va.pid else b_ad

        st.success(f"Kazanan: {kazanan}")
        c1, c2 = st.columns(2)
        c1.metric(a_ad, f"{ra.accuracy:.0%}", f"{ra.avg_tokens:.0f} token")
        c2.metric(b_ad, f"{rb.accuracy:.0%}", f"{rb.avg_tokens:.0f} token")
        st.caption("Bu karşılaştırma sürüm geçmişine kaydedildi.")
