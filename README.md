# Yapay_Zeka_Destekli_Goruntu_Analizi_ve_Nesne_Tabanli_Bilgi_Getirme_Sistemi

Kullanıcının verdiği bir görüntüdeki ön plandaki nesneyi tespit eder, nesneye ait
anahtar kelimeler üretir ve bu bilgilerle web'den ilgili içerikleri özetleyerek
kullanıcıya sunar.

## Mimari

- **OpenCLIP** (önceden eğitilmiş, yeniden eğitilmemiş) — zero-shot sınıflandırma
  için özellik çıkarıcı olarak kullanıldı. Sınıflandırma başarımını artırmak için
  her nesne için birden fazla prompt (ör. "a backpack", "a school backpack",
  "a travel backpack") oluşturulup embedding'leri ortalanarak **prompt
  ensembling** uygulandı.
- **MLP Head** — CLIP özellikleri üzerine sıfırdan eğitilen çok katmanlı algılayıcı
  (Adam, 60 epoch, **linear probing**). CLIP'in güven değeri düşük olduğunda
  ikinci aşama doğrulama/destek modeli olarak devreye giriyor. (İki modeli
  ensemble etmek yerine bu yöntem tercih edildi çünkü CLIP tek başına zaten
  güçlü sonuçlar veriyordu.)
- Nesne belirlendikten sonra **WordNet** ile anahtar kelime üretimi, **DuckDuckGo**
  üzerinden web araması ve **TF-IDF** ile metin özetleme uygulanıyor.
- Sonuçlar JSON olarak kaydediliyor.

## Klasör Yapısı

\```
notebooks/    -> Colab notebook'ları
weights/      -> linear_head_weights.pth, class_names.json
report/       -> Proje raporu (PDF)
\```

## Notebook'lar

- [Ana Proje](COLAB_LINKINI_BURAYA_KOY) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINKINI_BURAYA_KOY)
- [Extra Model Eğitimi](COLAB_LINKINI_BURAYA_KOY) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](COLAB_LINKINI_BURAYA_KOY)

## Çalıştırmadan Önce

⚠️ `weights/linear_head_weights.pth` ve `weights/class_names.json` dosyalarını
projeye eklemeyi unutmayın — bunlar olmadan sistem çalışmaz.

## Kurulum

\```bash
pip install -r requirements.txt
\```

## Karşılaşılan Problemler ve Çözümler

- RGBA/gri tonlamalı görüntüler → RGB'ye dönüştürme
- Benzbirbirine benzeyen nesnelerin karışması → MLP destekli ikinci aşama sınıflandırma
- Web sonuçlarının uzun/reklam içerikli olması → filtreleme + TF-IDF özetleme
