# CLAUDE.md — RSNA Pediatric Bone Age (2017) Projesi

## 1. Proje Özeti
El röntgeni (sol el, X-ray) görüntülerinden çocuğun **kemik yaşını (bone age, ay cinsinden)** tahmin eden bir derin öğrenme modeli.
- Problem tipi: **Regresyon** (çıktı = tek bir sayı, ay)
- Girdiler: Görüntü + cinsiyet (`male`: True/False)
- Ana metrik: **MAE (Mean Absolute Error), ay cinsinden**
- Referans: 2017 yarışmasının birincisi yaklaşık **4.3 ay MAE** elde etti.
- Amaç: Portföy projesi + derin öğrenmeye geçiş için öğrenme projesi.

## 2. Kullanıcıyla Çalışma Kuralları (ÇOK ÖNEMLİ)
- **Açıklamalar her zaman Türkçe** olsun. İngilizce teknik terim kullanınca parantez içinde kısaca Türkçesini yaz (ör. "overfitting (aşırı öğrenme)").
- **Kod her zaman İngilizce**: değişken, fonksiyon, sütun adları, yorumlar İngilizce.
- **Sadece `.ipynb` notebook kullan, `.py` dosyası YOK** (proje Google Colab'e yüklenecek). Ortak kod `notebooks/00_common.ipynb` içinde; diğer notebook'lar onu `%run 00_common.ipynb` ile yükler.
- **Kodu doğrudan proje dosyalarına yaz** (mesajda tüm kodu tekrar etmeye gerek yok, önemli kısımları özetle).
- **Claude her şeyi kendisi yapar**: Fazları sırayla, kullanıcıyı beklemeden uygular; kodu yazar, yerelde çalıştırır (yerel GPU: RTX 4070 8GB), sonuçları kontrol eder ve sonraki faza geçer. Sadece gerçekten engel olan durumlarda (ör. Kaggle API anahtarı) kullanıcıya sorar. Sonunda yapılanları ve sonuçları Türkçe özetler.
- **Gereksiz mühendislikten kaçın**: En basit çalışan çözümü yaz. PyTorch Lightning, Hydra, MLflow, Docker, karmaşık class hiyerarşileri vb. KULLANMA. Basit config hücresi + düz fonksiyonlar yeterli.
- Her yeni kavramı (ör. transfer learning, data augmentation, learning rate scheduler) ilk kullanıldığında 2-3 cümleyle Türkçe açıkla.
- **Git commit hatırlatması**: Her anlamlı adımdan sonra "Şimdi commit atmanın tam zamanı" de ve bir commit mesajı öner (İngilizce, ör. `feat: add dataset class and augmentations`).
- Hata çıktığında önce hatanın ne anlama geldiğini Türkçe açıkla, sonra düzeltmeyi ver.

## 3. Ortam
- Varsayılan çalışma ortamı: **Google Colab (GPU: T4)**. Notebook'lar hem Colab'de hem yerelde çalışacak şekilde yazılsın (yollar `00_common.ipynb` içindeki config bölümünden ayarlanır). Colab'de proje Drive'da `MyDrive/bone-age/` altında; ham görüntüler hız için `/content/data`'ya indirilir, `splits.csv` ve `outputs/` Drive'da kalır.
- Yerelde (Windows) notebook içinde tanımlı Dataset sınıfı DataLoader worker süreçlerine gönderilemediği için `NUM_WORKERS = 0` kullanılır; Colab'de 2.
- Kütüphaneler: `torch`, `torchvision`, `timm`, `albumentations`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `tqdm`, `kaggle`
- Mixed precision (`torch.amp`) kullan → T4'te hız ve bellek kazancı.
- Sabit seed (`seed=42`) ile tekrarlanabilirlik sağla.

## 4. Veri
- Kaynak: Kaggle dataset `kmader/rsna-bone-age`
  - İndirme: `kagglehub.dataset_download("kmader/rsna-bone-age")` (`01_setup_data.ipynb`)
- Kullanılacak dosyalar:
  - `boneage-training-dataset.csv` → sütunlar: `id`, `boneage` (ay), `male` (bool)
  - `boneage-training-dataset/` → `{id}.png` görüntüleri (~12.6k adet)
- Kaggle'daki test CSV'sinde etiket (boneage) **yok**. Bu yüzden eğitim setini kendimiz bölüyoruz:
  - **%80 train / %10 validation / %10 test**
  - Bölme, `boneage` aralıkları (bin) + `male` birlikte **stratify** edilerek yapılsın.
  - Bölme bir kez yapılıp `data/splits.csv` olarak kaydedilsin; tüm deneyler aynı bölmeyi kullansın.
- **Test seti sadece en sonda, bir kez** değerlendirilir. Model seçimi ve ayar sadece validation ile yapılır (data leakage / veri sızıntısı önlemi).

## 5. Klasör Yapısı
```
bone-age/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/                           # (git'e eklenmez) ham veri + görüntü cache; splits.csv
├── notebooks/
│   ├── 00_common.ipynb             # config + tüm ortak fonksiyonlar (dataset, model, train, evaluate, grad-cam)
│   ├── 01_setup_data.ipynb         # Faz 0: kurulum, veri indirme, görüntü cache
│   ├── 02_eda.ipynb                # Faz 1
│   ├── 03_split_baseline.ipynb     # Faz 2
│   ├── 04_dataset_model_check.ipynb# Faz 3 + 4
│   ├── 05_train.ipynb              # Faz 5
│   ├── 06_experiments.ipynb        # Faz 7 (sadece validation)
│   ├── 07_test_results.ipynb       # Faz 6 (test seti bir kez, en sonda)
│   └── 08_gradcam.ipynb            # Faz 8
├── outputs/                        # (git'e eklenmez) checkpoint, log, grafikler
└── .gitignore
```

## 6. Fazlar (sırayla, tek tek)

### Faz 0 — Kurulum
- Klasör yapısı, `requirements.txt`, `.gitignore` (`data/`, `outputs/`, `*.pth`, `kaggle.json`)
- Colab için: Drive bağlama, Kaggle API ile veri indirme hücreleri
- ✅ Çıktı: Veri indirildi, bir görüntü açılıp gösterildi.

### Faz 1 — EDA (Keşifçi Veri Analizi)
- `boneage` dağılımı (histogram), cinsiyete göre dağılım (boxplot)
- Cinsiyet dengesi
- Görüntü boyutları dağılımı (genişlik/yükseklik), örnek görüntüler (farklı yaşlardan grid)
- Görüntü kalitesi farklılıkları (parlaklık, kontrast, eğiklik) hakkında gözlemler
- ✅ Çıktı: `notebooks/02_eda.ipynb` + kısa Türkçe bulgular listesi

### Faz 2 — Veri Bölme + Baseline
- `03_split_baseline.ipynb`: stratified split → `data/splits.csv`
- **Baseline (temel model)**:
  1. Herkese train ortalamasını tahmin et → MAE
  2. Cinsiyete göre ortalama tahmin et → MAE
- Bu sayılar, derin öğrenme modelinin "gerçekten öğrenip öğrenmediğini" görmek için alt sınırdır.
- ✅ Çıktı: Baseline MAE değerleri

### Faz 3 — Dataset ve Augmentation
- `00_common.ipynb` (kontrol: `04_dataset_model_check.ipynb`): PyTorch `Dataset` sınıfı → (image, sex, target) döndürür
- Görüntüler gri tonlamalı okunur, 3 kanala kopyalanır (pretrained model için)
- Ön işleme: Resize (başlangıç **384x384**, en-boy oranı korunarak padding), ImageNet normalize
- Augmentation (sadece train): küçük rotasyon (±15°), shift/scale, parlaklık/kontrast, CLAHE (opsiyonel)
- Hedef değişken normalize edilir: `(boneage - mean) / std` (train istatistikleri ile); tahminde geri çevrilir
- Augmentation örneklerini görselleştiren küçük bir fonksiyon ekle
- ✅ Çıktı: Bir batch görselleştirildi, şekiller (shape) doğru

### Faz 4 — Model
- `00_common.ipynb` (kontrol: `04_dataset_model_check.ipynb`):
  - Backbone: `timm` ile pretrained **EfficientNet-B0** (alternatif: ResNet50)
  - Cinsiyet girdisi: küçük bir `Linear(1, 16)` katmanı
  - Görüntü özellikleri + cinsiyet özellikleri birleştirilir (concat) → `MLP` → 1 çıktı
- Transfer learning (aktarım öğrenimi) mantığını açıkla
- ✅ Çıktı: Rastgele bir batch ile forward pass çalışıyor

### Faz 5 — Eğitim
- `05_train.ipynb`:
  - Loss: `L1Loss` (doğrudan MAE'yi optimize eder)
  - Optimizer: `AdamW`, lr=1e-3 (head) / 1e-4 (backbone) veya tek lr ile başla
  - Scheduler: `CosineAnnealingLR` (veya `ReduceLROnPlateau`)
  - Epoch: ~20-30, early stopping (patience=5, validation MAE'ye göre)
  - En iyi modeli `outputs/best_model.pth` olarak kaydet
  - Her epoch: train loss, val MAE (ay cinsinden) yazdır ve CSV log'a kaydet
- ✅ Çıktı: Öğrenme eğrileri (loss/MAE vs epoch)

### Faz 6 — Değerlendirme
- `07_test_results.ipynb` (sadece en iyi model, **test seti bir kez**):
  - Test MAE (ay), RMSE, R²
  - Tahmin vs gerçek scatter plot (y=x çizgisiyle)
  - Hata dağılımı histogramı
  - Yaş grubuna göre MAE (ör. 0-5, 5-10, 10-15, 15+ yaş)
  - Cinsiyete göre MAE
  - En kötü 10 tahmin → görüntüleriyle birlikte göster ve yorumla
- ✅ Çıktı: `notebooks/07_test_results.ipynb`

### Faz 7 — İyileştirmeler (opsiyonel, birer birer dene)
Her denemeyi tek değişiklikle yap ve val MAE'yi tabloya kaydet:
1. Görüntü boyutunu 512'ye çıkarmak
2. Daha büyük backbone (EfficientNet-B3)
3. TTA (Test Time Augmentation)
4. Birkaç modelin ortalaması (ensemble)

### Faz 8 — Açıklanabilirlik (Grad-CAM)
- Modelin elin hangi bölgelerine baktığını ısı haritası ile göster (büyüme plakaları, bilek kemikleri beklenir)
- ✅ Çıktı: Farklı yaşlardan örnekler için Grad-CAM görselleri

### Faz 9 — Portföy için README
- Problem, veri, yöntem, sonuç tablosu (baseline vs model vs yarışma birincisi), grafikler, sınırlamalar
- README İngilizce yazılsın (portföy için), kullanıcıya Türkçe özet de verilsin.

## 7. Başarı Hedefleri
| Seviye | Test MAE |
|---|---|
| Baseline (ortalama) | ~30+ ay |
| İlk çalışan model | < 10 ay |
| İyi sonuç | < 6 ay |
| Yarışma birincisi | ~4.3 ay |

## 8. Kod Kuralları
- Tüm hiperparametreler ve yollar `00_common.ipynb` config bölümünde (basit sabitler)
- Fonksiyonlar kısa ve tek iş yapan; gereksiz soyutlama yok
- `tqdm` ile ilerleme çubuğu
- Her faz notebook'u ilk hücrede (Colab'deyse) Drive'ı bağlar, `notebooks/` klasörüne geçer ve `%run 00_common.ipynb` çalıştırır → tek başına çalıştırılabilir
- `00_common.ipynb` sadece tanım içerir (eğitim/çıktı üretmez)
- Uzun eğitimlerde checkpoint'i Google Drive'a kaydet (Colab oturumu kopabilir)