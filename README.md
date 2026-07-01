# 🦠 UAS Data Mining — COVID-19 Risk Prediction + Contact Tracing Graph

**Kelompok 1 | Prediksi Risiko Penyebaran COVID-19 dengan KNN + Contact Tracing Graph**

## 📋 Deskripsi
Dashboard interaktif untuk UAS Data Mining menggunakan metodologi CRISP-DM:
- **KNN Classification** — Prediksi risiko COVID-19 (Elevated/Low Risk)
- **Contact Tracing Graph** — Analisis jaringan kontak siswa sekolah
- **Graph Analytics** — Degree Centrality, Betweenness Centrality, Community Detection

## 🚀 Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy ke Streamlit Cloud

1. Push ke GitHub (repository public)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → pilih `app.py` → Deploy

## 📊 Dataset

| Dataset | Sumber | Keterangan |
|---|---|---|
| COVID-19 Symptoms & Severity | [Kaggle khushikyad001](https://www.kaggle.com/datasets/khushikyad001/covid-19-symptoms-and-severity-prediction-dataset) | Dataset 1 (KNN) |
| SocioPatterns High School | [SocioPatterns Thiers13](http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/) | Dataset 2 (Graph) |

## 👥 Anggota Kelompok 1
- Enjing Suandi
- Galang Rivaldi
- Aldafa Rayhandika

## 🏫 Framework
CRISP-DM (Cross-Industry Standard Process for Data Mining)
