# Yapay Zeka Destekli Goruntu Analizi ve Nesne Tabanli Bilgi Getirme Sistemi

Kullanıcının verdiği bir görüntüdeki ön plandaki nesneyi tespit eder, nesneye ait anahtar kelimeler üretir ve bu bilgilerle web'den ilgili içerikleri özetleyerek kullanıcıya sunar.

## Mimari

- **OpenCLIP** (önceden eğitilmiş, yeniden eğitilmemiş) — zero-shot sınıflandırma için özellik çıkarıcı olarak kullanıldı. Prompt ensembling uygulandı.
- **MLP Head** — CLIP özellikleri üzerine sıfırdan eğitilen çok katmanlı algılayıcı (Adam, 60 epoch, linear probing). CLIP'in güven değeri düşük olduğunda ikinci aşama doğrulama modeli olarak devreye giriyor.
- WordNet ile anahtar kelime üretimi, DuckDuckGo web araması, TF-IDF özetleme.

## Klasör Yapısı

notebooks/ -> Colab notebook'ları
weights/ -> linear_head_weight.pth, class_names.json
report/ -> Proje raporu (PDF)

## Notebook'lar

- Ana Proje: https://colab.research.google.com/drive/1lPk0HX-qPJMAs_MDZxqWF3hCjtiC7jxC?usp=sharing
- Extra Model Eğitimi: https://colab.research.google.com/drive/17HSL_chC8qFA1NjKEr4Rx9W3OydoWIjn?usp=sharing

## Çalıştırmadan Önce

weights/linear_head_weights.pth ve weights/class_names.json dosyalarını eklemeyi unutmayın.

## Kurulum

pip install -r requirements.txt