# Prompt Optimizasyon Platformu

Bir görev için farklı prompt mühendisliği stratejilerini otomatik olarak deneyen,
bunları aynı koşullarda karşılaştıran ve en iyi performans gösteren stratejiyi seçen
deneysel bir platform. Yüksek lisans tez çalışması kapsamında, prompt mühendisliği
tekniklerinin kontrollü karşılaştırması için geliştirilmiştir.

## Canlı Demo

Uygulama Docker tabanlı olarak Dokploy üzerinde yayınlanmaktadır. Yayındaki sürüm,
API anahtarı gerektirmeyen deneme (mock) modunda çalışır: arayüzün tümü ve karşılaştırma
akışı denenebilir, sonuçlar temsilî (simülasyon) verilerle gösterilir. Gerçek model
sonuçları için bir sağlayıcı (DeepSeek/GLM) anahtarı tanımlanması yeterlidir.

> Canlı adres: https://prompt-optimization-platform.berkayderin.com.tr/

## Motivasyon

Literatürde Chain-of-Thought, ReAct, Tree-of-Thoughts gibi çok sayıda prompt tekniği
ve OPRO, APE, DSPy gibi otomatik optimizasyon yöntemleri ayrı ayrı önerilmiştir. Ancak
bu teknikler çoğunlukla farklı veri setleri üzerinde, birbirinden bağımsız test edilmiş;
aynı görev kümesi üzerinde adil ve kontrollü bir karşılaştırma sunan, "bu görev için
hangi strateji daha iyi?" sorusunu yanıtlayan bütünleşik bir araç eksik kalmıştır. Bu
platform, söz konusu boşluğu doldurmayı hedefler.

## Özellikler

- Tek bir görev setinde altı farklı prompt stratejisinin kontrollü karşılaştırması
- Dört metrik: doğruluk, tutarlılık (aynı soru N kez sorulup yanıt kararlılığı ölçülür),
  ortalama token maliyeti ve gecikme süresi
- Kullanıcının kendi görevini (soru + beklenen cevap) girip tüm stratejileri kendi
  görevi üzerinde deneyebilmesi
- Açık uçlu görevlerde (özetleme) LLM-as-judge ile puanlama; kapalı uçlu görevlerde
  kural tabanlı eşleşme
- Kazanan stratejinin neden öne çıktığını sade dille açıklayan sonuç yorumu
- Meta-prompting tabanlı otomatik prompt iyileştirme (iteratif döngü, skor
  artmayınca erken durdurma)
- İki stratejiyi doğrudan kıyaslayan ve sonucu zaman damgasıyla sürüm geçmişine
  kaydeden A/B testi; kayıtların arayüzdeki **Geçmiş** sekmesinden izlenmesi
- Tüm deneyleri tek komutla koşup `data/results/` altına kaydeden betik
  (`run_experiments.py`) — makale tabloları bu çıktılardan beslenir
- Sağlayıcı bağımsız mimari: DeepSeek, GLM, OpenAI ve yerel Ollama desteği
- API anahtarı gerektirmeyen deneme (mock) modu ile tüm akışın çevrimdışı denenebilmesi
- Konuya yabancı kullanıcılar için adım adım yönlendirmeli arayüz
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
app.py                  Streamlit arayüzü (beş sekme: karşılaştırma, kendi göreviniz,
                        optimizasyon, A/B testi, geçmiş)
run_experiments.py      Tüm deneyleri komut satırından koşup sonuçları kaydeden betik
core/
  llm_client.py         OpenAI uyumlu tek istemci (mock/deepseek/glm/openai/ollama)
  strategies.py         Prompt stratejilerinin şablonları
  datasets.py           Görev setlerini JSON dosyalarından yükleme
  evaluator.py          Stratejileri çalıştırma; doğruluk/tutarlılık/token/gecikme
                        skorlama (kural tabanlı + LLM-as-judge)
  optimizer.py          Meta-prompting ile otomatik iyileştirme (erken durdurmalı)
  versioning.py         Prompt sürümleme (zaman damgalı) ve A/B karşılaştırma
data/
  tasks/                Görev setleri (aritmetik, duygu analizi, konu sınıflandırma,
                        özetleme)
  results/              Deney çıktıları: sürüm geçmişi (versions.json), karşılaştırma
                        ve deney CSV/JSON dosyaları
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
tanımlanmadığında deneme (mock) modda çalışır; arayüz ve tüm akış denenebilir, ancak
doğruluk değerleri temsilidir.

## Nasıl Kullanılır

Arayüz, konuya yabancı kullanıcılar için adım adım yönlendirilmiştir:

1. Soldaki panelden bir **görev seti** seçin (aritmetik, duygu analizi, konu
   sınıflandırma veya özetleme). Seçilen setin ne olduğu ve neden seçildiği ekranda
   açıklanır. İsterseniz sıcaklığı ve tutarlılık ölçümü için tekrar sayısını (N)
   ayarlayın.
2. Bir sekme açın:
   - **Karşılaştırma**: Tüm stratejileri aynı sette dener ve en iyisini, ardından
     bu sonucun nedenini açıklar. Başlamak için en uygun seçenek budur.
   - **Kendi Göreviniz**: Kendi sorularınızı ve beklenen cevaplarını girersiniz;
     sistem tüm stratejileri sizin göreviniz üzerinde dener ve en iyisini seçer.
   - **Optimizasyon**: Tek bir başlangıç talimatını tur tur otomatik iyileştirir;
     skor artmazsa erken durur.
   - **A/B Testi**: Seçtiğiniz iki yöntemi birebir karşılaştırır ve sonucu zaman
     damgasıyla sürüm geçmişine kaydeder.
   - **Geçmiş**: Kaydedilen tüm deneyleri (hangi strateji, hangi set, hangi
     ayarlar, hangi sonuç) listeler.
3. **Çalıştır** düğmesine basın ve sonuçları görün.

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
(`arithmetic`, `classification` veya `summarization`), few-shot örnekleri ve
`question`/`answer` çiftlerinden oluşan görev listesi içerir. Yeni bir set eklemek için
bu klasöre aynı yapıda bir JSON dosyası eklemek yeterlidir; arayüz dosyayı otomatik tanır.

## Değerlendirme

Tüm stratejiler aynı görev seti üzerinde çalıştırılır. Puanlama görev tipine göre
yapılır:

- **Aritmetik**: yanıttaki son sayı çıkarılır ve beklenen sayıyla karşılaştırılır
  (exact match).
- **Sınıflandırma**: beklenen etiketin yanıt metninde geçmesi yeterlidir.
- **Özetleme (açık uçlu)**: tek bir doğru cevap olmadığından model hakem olarak
  kullanılır (LLM-as-judge): yanıtın referans özetle özünde aynı bilgiyi taşıyıp
  taşımadığına sıcaklık 0 ile karar verilir. Hakem çağrılarının maliyeti, stratejinin
  kendi maliyetini ölçen token/gecikme metriklerine dahil edilmez.

Değerlendirme motoru her strateji için dört metrik hesaplar: **doğruluk**,
**tutarlılık** (her soru N kez sorulduğunda çalıştırmalar arası kararlılık; soru
başına çoğunluk oranının ortalaması, N=1'de tanım gereği 1.0), **ortalama token**
ve **ortalama gecikme**.

Deneme (mock) modunda gerçek model çağrısı yapılmaz; bunun yerine her strateji için
deterministik, temsilî sonuçlar üretilir. Bu sayede platform anahtarsız olarak
denenebilir. Bu değerler bilimsel ölçüm değildir; gerçek sonuçlar bir sağlayıcı
tanımlanarak elde edilir.

## Deneylerin Komut Satırından Koşulması

Makaleyi besleyecek deneyler tek komutla koşulur; tüm görev setlerinde tüm stratejiler
çalıştırılır, sonuçlar `data/results/` altına CSV (set başına) ve JSON özeti olarak
kaydedilir:

```bash
python run_experiments.py                      # .env'deki sağlayıcı ile
python run_experiments.py --provider deepseek  # sağlayıcıyı elle seç
python run_experiments.py --runs 3             # tutarlılık için her soruyu 3 kez sor
```

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
