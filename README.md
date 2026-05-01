# 🏦 Prediksi Estimasi Beban Pajak (PPh/PPN)

Proyek ML untuk memprediksi beban pajak kuartal mendatang berdasarkan tren penjualan historis.

## Struktur File
```
tax_prediction/
├── tax_pipeline.py      # Pipeline utama: Ingestion → Preprocessing → EDA → Model → Evaluasi
├── app_dashboard.py     # Dashboard Streamlit interaktif
├── requirements.txt     # Daftar library Python
└── README.md
```

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan pipeline ML (terminal / Jupyter)
```bash
python tax_pipeline.py
```
Output: plot EDA & evaluasi disimpan di folder `output_plots/`

### 3. Jalankan Dashboard Streamlit
```bash
streamlit run app_dashboard.py
```
Buka browser → `http://localhost:8501`

## Alur Pipeline

| #  | Tahap              | File              | Keterangan                                      |
|----|--------------------|-------------------|-------------------------------------------------|
| 1  | Data Ingestion     | tax_pipeline.py   | Simulasi ERP → DataFrame dengan dirty data      |
| 2  | Preprocessing      | tax_pipeline.py   | Dedup, imputasi NaN, hapus outlier IQR, scaler  |
| 3  | EDA                | tax_pipeline.py   | Statistik deskriptif, heatmap, tren kuartal     |
| 4  | Modeling           | tax_pipeline.py   | Linear Regression (fitur: Penjualan, HPP, Ops)  |
| 5  | Evaluasi           | tax_pipeline.py   | MAE, RMSE, R², plot aktual vs prediksi          |
| 6  | Deployment         | app_dashboard.py  | Dashboard Streamlit interaktif dengan slider    |

## Catatan Perpajakan
- **PPh Badan**: 22% × Laba Kena Pajak (UU HPP 2022)
- **PPN Keluaran**: 11% × Penjualan Bersih (UU HPP 2022)
- Model dilatih untuk memprediksi **total beban pajak** secara langsung
