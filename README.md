# Prompt Optimizasyon Platformu

Bir görev için farklı prompt mühendisliği stratejilerini otomatik olarak deneyen,
bunları aynı koşullarda karşılaştıran ve en iyi performans gösteren stratejiyi seçen
deneysel bir platform. Yüksek lisans tez çalışması kapsamında, prompt mühendisliği
tekniklerinin kontrollü karşılaştırması için geliştirilmiştir.

## Motivasyon

Literatürde Chain-of-Thought, ReAct, Tree-of-Thoughts gibi çok sayıda prompt tekniği
ve OPRO, APE, DSPy gibi otomatik optimizasyon yöntemleri ayrı ayrı önerilmiştir. Ancak
bu teknikler çoğunlukla farklı veri setleri üzerinde, birbirinden bağımsız test edilmiş;
aynı görev kümesi üzerinde adil ve kontrollü bir karşılaştırma sunan, "bu görev için
hangi strateji daha iyi?" sorusunu yanıtlayan bütünleşik bir araç eksik kalmıştır. Bu
platform, söz konusu boşluğu doldurmayı hedefler.

## Özellikler

- Tek bir görev setinde altı farklı prompt stratejisinin kontrollü karşılaştırması
- Doğruluk, ortalama token maliyeti ve ortalama gecikme metrikleri
- Meta-prompting tabanlı otomatik prompt iyileştirme (OPRO benzeri iteratif döngü)
- İki stratejiyi doğrudan kıyaslayan A/B testi
- Sağlayıcı bağımsız mimari: DeepSeek, GLM, OpenAI ve yerel Ollama desteği
- API anahtarı gerektirmeyen mock mod ile tüm akışın çevrimdışı denenebilmesi
- Sonuçların CSV olarak dışa aktarılması

## Desteklenen Stratejiler

| Strateji | Açıklama |
|---|---|
| Zero-shot | Örneksiz, doğrudan talimat (temel çizgi) |
| Few-shot | Bağlam içi öğrenme için çözülmüş örnekler |
| Chain-of-Thought | Cevaptan önce adım adım muhakeme |
| ReAct | Düşünce-Eylem-Gözlem döngüsü |
| Tree-of-Thoughts | Birden çok çözüm yolu üretip seçme |
| Meta-prompting | Modelin görev için kendi talimatını tasarlaması |

## Mimari

```
app.py                  Streamlit arayüzü (üç sekme: karşılaştırma, optimizasyon, A/B)
core/
  llm_client.py         OpenAI uyumlu tek istemci (mock/deepseek/glm/openai/ollama)
  strategies.py         Prompt stratejilerinin şablonları
  datasets.py           Görev setlerini JSON dosyalarından yükleme
  evaluator.py          Stratejileri çalıştırma ve metriklerle skorlama
  optimizer.py          Meta-prompting ile otomatik iyileştirme
  versioning.py         Prompt sürümleme ve A/B karşılaştırma
data/
  tasks/                Görev setleri (aritmetik, duygu analizi, konu sınıflandırma)
  results/              Deney çıktıları (sürüm geçmişi, dışa aktarmalar)
```

DeepSeek, GLM, OpenAI ve Ollama hepsi OpenAI uyumlu bir API sunduğundan, yalnızca
`base_url` ve `api_key` değiştirilerek tek bir istemci üzerinden kullanılır.

## Kurulum

Gereksinim: Python 3.9 veya üzeri.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır. Anahtar
tanımlanmadığında mock modda çalışır; arayüz ve tüm akış denenebilir, ancak doğruluk
değerleri temsilidir.

## Yapılandırma

`.env.example` dosyasını `.env` olarak kopyalayın ve sağlayıcıyı seçin:

```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
```

| Sağlayıcı | Ortam değişkeni | Örnek model |
|---|---|---|
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` |
| GLM (Zhipu) | `GLM_API_KEY` | `glm-4-flash`, `glm-4-plus` |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Ollama (yerel) | gerekmez | `llama3` |

## Görev Setleri

Görev setleri `data/tasks/` altında JSON olarak tanımlanır. Her set bir ad, görev tipi
(`arithmetic` veya `classification`), few-shot örnekleri ve `question`/`answer`
çiftlerinden oluşan görev listesi içerir. Yeni bir set eklemek için bu klasöre aynı
yapıda bir JSON dosyası eklemek yeterlidir; arayüz dosyayı otomatik tanır.

## Değerlendirme

Tüm stratejiler aynı görev seti üzerinde çalıştırılır. Model yanıtından cevap çıkarılır
(sayısal görevlerde son sayı, sınıflandırma görevlerinde beklenen etiketin metinde
geçmesi) ve beklenen cevapla karşılaştırılır. Her strateji için doğruluk oranı, ortalama
token sayısı ve ortalama gecikme raporlanır.

## Docker ile Çalıştırma

```bash
docker build -t prompt-platform .
docker run -p 8501:8501 -e LLM_PROVIDER=mock prompt-platform
```

## Dokploy ile Dağıtım

1. Dokploy panelinde yeni bir Application oluşturun ve bu GitHub deposunu bağlayın.
2. Build tipini Dockerfile olarak seçin.
3. Ortam değişkenlerini (`LLM_PROVIDER`, gerekli API anahtarları) tanımlayın.
4. Container portunu `8501` olarak ayarlayıp bir alan adı bağlayın ve dağıtın.

Dockerfile, `/_stcore/health` uç noktasını kontrol eden bir sağlık kontrolü içerir.

## Lisans

Bu çalışma akademik amaçlıdır.
