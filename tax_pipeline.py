# -*- coding: utf-8 -*-
"""
=============================================================================
  PIPELINE PREDIKSI ESTIMASI BEBAN PAJAK (PPh / PPN) — KUARTAL MENDATANG
=============================================================================
  Alur Kerja (6 Tahap):
    1. Data Ingestion     → Simulasi ekstraksi ERP (CSV / JSON / SQL)
    2. Preprocessing      → Cleaning · Dedup · Outlier · Normalisasi
    3. EDA                → Statistik deskriptif + Visualisasi
    4. Modeling           → Linear Regression
    5. Evaluasi           → MAE · RMSE · R²
    6. Export Artefak     → Pickle model + Simpan plot ke output_plots/

  Referensi Regulasi (UU HPP No.7 Tahun 2021, berlaku 2022):
    • PPh Badan     : 22% × Laba Kena Pajak
    • PPN Keluaran  : 11% × Penjualan Bersih

  Cara menjalankan:
    python tax_pipeline.py
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORT LIBRARY & KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import sys
import warnings

# Paksa stdout ke UTF-8 agar emoji di print() tidak crash di Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

import matplotlib

# Gunakan backend 'Agg' (non-interactive) saat berjalan sebagai script CLI
# agar plt.show() tidak memblokir eksekusi dan plot disimpan langsung ke file.
# Saat diimpor oleh Streamlit, backend sudah di-set oleh Streamlit sehingga
# baris ini tidak akan mengubah perilaku dashboard.
if not any("streamlit" in arg for arg in sys.argv):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")  # Sembunyikan peringatan minor runtime
sns.set_theme(style="whitegrid", palette="muted")  # Tema visual seragam di semua plot

# ─── Konstanta Global ────────────────────────────────────────────────────────
OUTPUT_DIR = "output_plots"  # Folder penyimpanan gambar hasil EDA & evaluasi
MODEL_PATH = "model_artefak.pkl"  # Berkas artefak model (binary pickle)
RANDOM_SEED = 42  # Seed untuk reproduksibilitas eksperimen
TAX_PPH_RATE = 0.22  # Tarif PPh Badan — UU HPP 2022
TAX_PPN_RATE = 0.11  # Tarif PPN Keluaran — UU HPP 2022

# Kolom fitur (X) yang digunakan untuk melatih model
FEATURES = ["penjualan_bersih", "hpp", "beban_operasional"]
# Kolom target (y) yang ingin diprediksi oleh model
TARGET = "total_beban_pajak"

# Path file data pengguna (CSV yang diunggah / diinput manual)
USER_DATA_PATH = "data_keuangan.csv"

# Kolom minimal yang WAJIB ada di data input pengguna
REQUIRED_INPUT_COLS = [
    "tahun",
    "kuartal",
    "penjualan_bersih",
    "hpp",
    "beban_operasional",
]


# =============================================================================
# MODUL DATA PENGGUNA
# Fungsi untuk memuat, menyimpan, memvalidasi, dan menghitung kolom pajak
# dari data keuangan yang diinput langsung oleh tim Finance.
# =============================================================================


def hitung_kolom_pajak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung kolom-kolom pajak turunan dari 3 kolom input dasar.

    Kolom INPUT yang wajib ada (dari pengguna):
      tahun             → Tahun fiskal (int)
      kuartal           → Nomor kuartal 1–4 (int)
      penjualan_bersih  → Pendapatan bersih (Rp)
      hpp               → Harga Pokok Penjualan (Rp)
      beban_operasional → Beban operasional (Rp)

    Kolom OUTPUT yang dihitung otomatis:
      periode           → Label "2023-Q1" dst.
      ppn_keluaran      → 11% × penjualan_bersih
      pph_badan         → 22% × max(0, laba_sebelum_pajak)
      total_beban_pajak → ppn_keluaran + pph_badan

    Args:
        df: DataFrame dengan minimal kolom REQUIRED_INPUT_COLS.

    Returns:
        pd.DataFrame: DataFrame lengkap dengan semua kolom pajak.
    """
    df = df.copy()

    # Pastikan tipe data numerik benar
    for col in ["tahun", "kuartal", "penjualan_bersih", "hpp", "beban_operasional"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Generate label periode jika belum ada
    if "periode" not in df.columns:
        df["periode"] = df.apply(
            lambda r: (
                f"{int(r['tahun'])}-Q{int(r['kuartal'])}"
                if pd.notna(r["tahun"]) and pd.notna(r["kuartal"])
                else "Unknown"
            ),
            axis=1,
        )

    # Kalkulasi pajak sesuai UU HPP 2022
    df["ppn_keluaran"] = df["penjualan_bersih"] * TAX_PPN_RATE
    laba_kotor = df["penjualan_bersih"] - df["hpp"]
    laba_sebelum_pajak = laba_kotor - df["beban_operasional"]
    df["pph_badan"] = (laba_sebelum_pajak * TAX_PPH_RATE).clip(lower=0)
    df["total_beban_pajak"] = df["ppn_keluaran"] + df["pph_badan"]

    # Urutkan berdasarkan waktu
    if "tahun" in df.columns and "kuartal" in df.columns:
        df = df.sort_values(["tahun", "kuartal"]).reset_index(drop=True)

    return df


def validasi_input_data(df: pd.DataFrame) -> list:
    """
    Memvalidasi DataFrame input dari pengguna sebelum diproses.

    Pemeriksaan yang dilakukan:
      1. Kolom wajib harus ada
      2. Tidak ada baris yang seluruh kolomnya kosong
      3. Nilai numerik tidak boleh negatif
      4. Kuartal harus bernilai 1, 2, 3, atau 4
      5. Minimal 8 baris data (agar train/test split bermakna)

    Args:
        df: DataFrame yang akan divalidasi.

    Returns:
        list: Daftar pesan error. Kosong berarti data valid.
    """
    errors = []

    # 1. Cek kolom wajib
    missing_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Kolom wajib tidak ditemukan: {missing_cols}")
        return errors  # Stop jika kolom tidak ada

    # 2. Cek baris semua-null
    _null_mask = df[REQUIRED_INPUT_COLS].isna().all(axis=1)
    all_null = int(np.array(_null_mask, dtype=int).sum())
    if all_null > 0:
        errors.append(f"{all_null} baris kosong sepenuhnya ditemukan.")

    # 3. Cek nilai negatif pada kolom numerik
    for col in ["penjualan_bersih", "hpp", "beban_operasional"]:
        if (df[col].dropna() < 0).any():
            errors.append(f"Kolom '{col}' mengandung nilai negatif.")

    # 4. Cek validitas kuartal
    invalid_q = df["kuartal"].dropna()
    invalid_q = invalid_q[~invalid_q.isin([1, 2, 3, 4])]
    if len(invalid_q) > 0:
        errors.append(
            f"Kolom 'kuartal' memiliki nilai tidak valid: {sorted(set(invalid_q.tolist()))}"
        )

    # 5. Cek jumlah baris minimal
    if len(df.dropna(subset=REQUIRED_INPUT_COLS)) < 8:
        errors.append(
            f"Data terlalu sedikit: {len(df)} baris. "
            f"Minimal 8 baris (2 tahun × 4 kuartal) diperlukan untuk melatih model."
        )

    return errors


def save_user_data(df: pd.DataFrame) -> None:
    """
    Menyimpan data keuangan pengguna ke file CSV.

    Hanya kolom INPUT (bukan kolom kalkulasi) yang disimpan agar file
    tetap bersih dan dapat diedit secara manual di Excel.

    Args:
        df: DataFrame yang berisi minimal REQUIRED_INPUT_COLS.
    """
    save_cols = [
        c
        for c in [
            "periode",
            "tahun",
            "kuartal",
            "penjualan_bersih",
            "hpp",
            "beban_operasional",
        ]
        if c in df.columns
    ]
    df[save_cols].to_csv(USER_DATA_PATH, index=False, encoding="utf-8")
    print(f"  ✔ Data pengguna disimpan ke: '{USER_DATA_PATH}'  ({len(df)} baris)")


def load_user_data() -> "pd.DataFrame | None":
    """
    Memuat data keuangan pengguna dari file CSV dan menghitung kolom pajak.

    Returns:
        pd.DataFrame: DataFrame lengkap dengan semua kolom pajak, atau
                      None jika file tidak ditemukan.
    """
    if not os.path.exists(USER_DATA_PATH):
        return None
    df = pd.read_csv(USER_DATA_PATH, encoding="utf-8")
    return hitung_kolom_pajak(df)


def buat_template_csv() -> str:
    """
    Membuat konten CSV template yang dapat diunduh oleh pengguna.

    Berisi contoh 8 baris data (2 tahun × 4 kuartal) dengan nilai
    placeholder yang realistis agar tim Finance dapat langsung mengisi.

    Returns:
        str: Konten CSV sebagai string UTF-8.
    """
    template = pd.DataFrame(
        {
            "tahun": [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
            "kuartal": [1, 2, 3, 4, 1, 2, 3, 4],
            "penjualan_bersih": [
                5_000_000_000,
                5_300_000_000,
                6_100_000_000,
                7_200_000_000,
                5_500_000_000,
                5_800_000_000,
                6_700_000_000,
                8_100_000_000,
            ],
            "hpp": [
                3_000_000_000,
                3_180_000_000,
                3_660_000_000,
                4_320_000_000,
                3_300_000_000,
                3_480_000_000,
                4_020_000_000,
                4_860_000_000,
            ],
            "beban_operasional": [
                750_000_000,
                795_000_000,
                915_000_000,
                1_080_000_000,
                825_000_000,
                870_000_000,
                1_005_000_000,
                1_215_000_000,
            ],
        }
    )
    return template.to_csv(index=False, encoding="utf-8")


# =============================================================================
# TAHAP 1 — DATA INGESTION
# Mensimulasikan proses ekstraksi data historis keuangan dari sistem ERP.
# =============================================================================


def generate_erp_data() -> pd.DataFrame:
    """
    Mensimulasikan ekstraksi data historis keuangan dari sistem ERP perusahaan.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Dalam proyek PRODUKSI, fungsi ini digantikan oleh salah satu dari:      │
    │   • Query SQL   : pd.read_sql("SELECT * FROM tb_keuangan", engine)      │
    │   • File CSV    : pd.read_csv("erp_keuangan_export.csv")               │
    │   • REST API    : requests.get("https://erp.corp.id/api/v1/finance")    │
    │   • File JSON   : pd.read_json("erp_export.json", orient="records")     │
    └─────────────────────────────────────────────────────────────────────────┘

    Kolom dataset yang dihasilkan:
      periode           → Label kuartal, cth. "2020-Q1"
      tahun             → Tahun fiskal (int)
      kuartal           → Nomor kuartal 1–4 (int)
      penjualan_bersih  → Pendapatan bersih setelah retur & diskon (Rp)
      hpp               → Harga Pokok Penjualan / COGS (Rp)
      beban_operasional → Biaya gaji, sewa, utilitas, pemasaran (Rp)
      ppn_keluaran      → PPN 11% × penjualan_bersih (Rp)
      pph_badan         → PPh Badan 22% × Laba Kena Pajak (Rp)
      total_beban_pajak → ppn_keluaran + pph_badan  ← TARGET PREDIKSI (Rp)

    Returns:
        pd.DataFrame: Dataset mentah (RAW) termasuk dirty data yang disengaja
                      untuk keperluan demonstrasi tahap Preprocessing.
    """
    np.random.seed(RANDOM_SEED)

    # ── Bangun sumbu waktu: 2015 Q1 s/d 2023 Q4 = 9 tahun × 4 kuartal = 36 baris ──
    tahun_list = [y for y in range(2015, 2024) for _ in range(4)]
    kuartal_list = [q for _ in range(2015, 2024) for q in range(1, 5)]
    n = len(tahun_list)  # Total observasi = 36 kuartal

    # ── Simulasi tren penjualan bersih ──────────────────────────────────────
    # Penjualan tumbuh ~9% per tahun (Rp 4 M → Rp 11 M)
    # Bobot musiman: Q1 rendah (awal tahun), Q4 tinggi (musim puncak)
    tren_dasar = np.linspace(4_000_000_000, 11_000_000_000, n)
    bobot_musiman = np.tile([0.82, 0.88, 1.05, 1.25], 9)  # Q1–Q4 per 9 tahun
    noise = np.random.normal(0, 250_000_000, n)  # Fluktuasi bisnis acak
    penjualan_bersih = (tren_dasar * bobot_musiman + noise).clip(min=1_000_000_000)

    # ── HPP: 55–65% dari penjualan (variatif tergantung efisiensi produksi) ─
    hpp = penjualan_bersih * np.random.uniform(0.55, 0.65, n)

    # ── Beban Operasional: 12–18% dari penjualan ────────────────────────────
    beban_operasional = penjualan_bersih * np.random.uniform(0.12, 0.18, n)

    # ── Kalkulasi pajak sesuai regulasi UU HPP 2022 ─────────────────────────
    ppn_keluaran = penjualan_bersih * TAX_PPN_RATE  # 11% × Penjualan
    laba_kotor = penjualan_bersih - hpp
    laba_sebelum_pajak = laba_kotor - beban_operasional
    # Clip ke 0: Tidak ada PPh jika perusahaan merugi (regulasi berlaku)
    pph_badan = (laba_sebelum_pajak * TAX_PPH_RATE).clip(min=0)
    total_beban_pajak = ppn_keluaran + pph_badan

    # ── Rakit DataFrame utama ───────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "periode": [f"{y}-Q{q}" for y, q in zip(tahun_list, kuartal_list)],
            "tahun": tahun_list,
            "kuartal": kuartal_list,
            "penjualan_bersih": penjualan_bersih.round(0),
            "hpp": hpp.round(0),
            "beban_operasional": beban_operasional.round(0),
            "ppn_keluaran": ppn_keluaran.round(0),
            "pph_badan": pph_badan.round(0),
            "total_beban_pajak": total_beban_pajak.round(0),
        }
    )

    # ── ⚠️  Injeksi "Dirty Data" — Simulasi kondisi data ERP nyata ─────────
    # (a) Missing values: 4 sel dikosongkan secara acak
    #     Contoh nyata: slip gaji belum diinput, laporan HPP terlambat, dll.
    dirty_indices = np.random.choice(df.index, size=4, replace=False)
    df.loc[dirty_indices[0], "beban_operasional"] = np.nan
    df.loc[dirty_indices[1], "hpp"] = np.nan
    df.loc[dirty_indices[2], "penjualan_bersih"] = np.nan
    df.loc[dirty_indices[3], "total_beban_pajak"] = np.nan

    # (b) Duplikat baris: 2 baris digandakan (terjadi ketika ETL berjalan dua kali)
    df = pd.concat([df, df.iloc[[3, 10]]], ignore_index=True)

    # (c) Outlier ekstrem: simulasi koreksi besar atau retur raksasa
    #     2017-Q3: penjualan sangat kecil (simulasi force-majeure)
    #     2020-Q1: total pajak sangat besar (simulasi koreksi audit COVID)
    df.loc[df["periode"] == "2017-Q3", "penjualan_bersih"] = 150_000_000
    df.loc[df["periode"] == "2020-Q1", "total_beban_pajak"] = 9_800_000_000

    print("=" * 70)
    print("  TAHAP 1 — DATA INGESTION")
    print("=" * 70)
    print(f"  ✔ Dataset ERP dibuat       : {df.shape[0]} baris × {df.shape[1]} kolom")
    print(
        f"  ✔ Rentang periode          : {df['periode'].dropna().iloc[0]}"
        f" → {df['periode'].dropna().iloc[-1]}"
    )
    print(f"  ✔ Missing values (dirty)   : {df.isnull().sum().sum()} sel")
    print(f"  ✔ Duplikat terinjeksi      : 2 baris")
    print(f"  ✔ Outlier terinjeksi       : 2 baris\n")

    return df


# =============================================================================
# TAHAP 2 — DATA PREPROCESSING
# Membersihkan dan mentransformasi data mentah agar siap untuk modeling.
# =============================================================================


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Menjalankan serangkaian langkah pembersihan dan transformasi data.

    Urutan proses internal:
      2.1 — Hapus baris duplikat
      2.2 — Imputasi missing values dengan nilai median
      2.3 — Deteksi & hapus outlier menggunakan metode IQR (Interquartile Range)
      2.4 — Normalisasi fitur numerik dengan StandardScaler

    Args:
        df (pd.DataFrame): Dataset mentah dari tahap Ingestion.

    Returns:
        tuple:
          • df_scaled    (pd.DataFrame)  : Data bersih dengan fitur ternormalisasi
          • scaler       (StandardScaler): Objek scaler yang telah di-fit (untuk inverse/transform)
          • df_raw_clean (pd.DataFrame)  : Data bersih SEBELUM scaling (untuk EDA & display)
    """
    print("=" * 70)
    print("  TAHAP 2 — DATA PREPROCESSING")
    print("=" * 70)

    df = df.copy()  # Jangan ubah DataFrame asli (immutability best practice)

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2.1 — Hapus Baris Duplikat
    # Duplikat muncul saat ETL pipeline dijalankan lebih dari sekali,
    # menyebabkan data tertentu terhitung ganda dalam analisis.
    # ─────────────────────────────────────────────────────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_dupl_removed = n_before - len(df)
    print(
        f"  [2.1] Duplikat dihapus    : {n_dupl_removed} baris → sisa {len(df)} baris"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2.2 — Tangani Missing Values
    # Strategi: Imputasi dengan nilai MEDIAN (robust terhadap outlier).
    # Alternatif yang dapat dipertimbangkan:
    #   • forward-fill (df.ffill()) untuk data time-series ketat
    #   • mean imputation jika distribusi normal dan tidak ada outlier
    #   • model-based imputation (IterativeImputer) untuk dataset besar
    # ─────────────────────────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if bool(df[col].isnull().any()):
            median_val = df[col].median()
            # Pandas 2.x: gunakan assignment langsung, bukan inplace=True
            # (fillna dengan inplace pada slice sudah deprecated di pandas 2.x)
            df[col] = df[col].fillna(median_val)
            print(
                f"  [2.2] Imputasi median     : '{col}' → median = Rp {median_val:,.0f}"
            )
    missing_remaining = df[numeric_cols].isnull().sum().sum()
    print(f"        Missing values sisa  : {missing_remaining} sel")

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2.3 — Hapus Outlier dengan Metode IQR
    # Rumus batas:
    #   Lower Fence = Q1 − 1.5 × IQR
    #   Upper Fence = Q3 + 1.5 × IQR
    # Data di luar rentang ini dianggap outlier dan dibuang.
    # Metode IQR dipilih karena:
    #   ✓ Non-parametrik (tidak mengasumsikan distribusi normal)
    #   ✓ Efektif mendeteksi "retur raksasa" atau "koreksi laporan besar"
    # ─────────────────────────────────────────────────────────────────────────
    cols_to_check = FEATURES + [TARGET]
    mask_valid = pd.Series(True, index=df.index)

    for col in cols_to_check:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        is_outlier = (df[col] < lower) | (df[col] > upper)
        if is_outlier.sum() > 0:
            print(
                f"  [2.3] Outlier terdeteksi  : '{col}' → {is_outlier.sum()} baris"
                f"  (fence: {lower:,.0f} – {upper:,.0f})"
            )
        mask_valid &= ~is_outlier  # Akumulasi mask: buang jika outlier di kolom manapun

    n_before_iqr = len(df)
    df = pd.DataFrame(df[mask_valid]).reset_index(drop=True)
    print(
        f"        Baris dibuang (IQR)  : {n_before_iqr - len(df)} → sisa {len(df)} baris"
    )

    # Simpan salinan data bersih SEBELUM scaling untuk keperluan EDA & tampilan
    df_raw_clean = df.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 2.4 — Normalisasi Fitur (StandardScaler)
    # Formula: z = (x − μ) / σ
    # Tujuan: Menyetarakan skala semua fitur agar model tidak "tertipu" oleh
    # fitur dengan nilai absolut lebih besar (misal: penjualan >> beban ops).
    #
    # CATATAN PENTING:
    #   • Scaler di-fit di sini pada SELURUH data bersih.
    #   • Dalam fungsi train_model(), scaler akan di-fit ULANG hanya pada
    #     data TRAINING untuk menghindari data leakage ke test set.
    #   • Objek scaler ini yang disimpan sebagai artefak untuk dashboard.
    # ─────────────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])  # type: ignore[index]

    print(f"\n  [2.4] StandardScaler diterapkan pada fitur: {FEATURES}")
    print(f"        Rata-rata (μ) setiap fitur (dalam Miliar Rp):")
    mean_vals = (
        np.array(scaler.mean_).flatten().tolist() if scaler.mean_ is not None else []
    )  # type: ignore[arg-type]
    for feat, mean_val in zip(FEATURES, mean_vals):
        print(f"          • {feat:<22}: Rp {mean_val:>15,.0f}")
    print(
        f"\n  ✔ Preprocessing selesai: {len(df_scaled)} baris data bersih & siap diproses.\n"
    )

    return df_scaled, scaler, df_raw_clean


# =============================================================================
# TAHAP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# Memahami pola, distribusi, dan korelasi dalam data yang sudah bersih.
# =============================================================================


def run_eda(df_raw_clean: pd.DataFrame, save_plots: bool = True) -> None:
    """
    Menjalankan analisis eksplorasi data (EDA) secara menyeluruh.

    Menghasilkan dua file plot:
      • eda_analysis.png            → 4-panel: heatmap, tren pajak, stacked area, boxplot
      • eda_scatter_penjualan.png   → Scatter: penjualan vs total pajak per kuartal

    Args:
        df_raw_clean (pd.DataFrame): DataFrame BERSIH (sebelum scaling) untuk visualisasi.
        save_plots   (bool)        : Jika True, simpan semua plot ke folder OUTPUT_DIR.
    """
    print("=" * 70)
    print("  TAHAP 3 — EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Buat folder output jika belum ada

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3.1 — Statistik Deskriptif
    # Memberikan gambaran umum: rata-rata, standar deviasi, kuartil, min, max.
    # ─────────────────────────────────────────────────────────────────────────
    analytic_cols = FEATURES + [TARGET, "ppn_keluaran", "pph_badan"]
    desc = df_raw_clean[analytic_cols].describe()

    print("\n  [3.1] STATISTIK DESKRIPTIF (dalam Miliar Rupiah):")
    print("-" * 70)
    # Hanya baris statistik numerik yang dikonversi ke miliar Rupiah
    # Baris 'count' tidak dikonversi karena bukan nilai moneter
    desc_display = desc.copy()
    numeric_stat_rows = [r for r in desc_display.index if r != "count"]
    desc_display.loc[numeric_stat_rows] = (
        desc_display.loc[numeric_stat_rows] / 1e9
    ).round(3)
    desc_display.loc["count"] = desc_display.loc["count"].astype(int)
    print(desc_display.to_string())
    print("-" * 70)

    # Insight keuangan tambahan yang berguna untuk tim Finance
    avg_pajak = df_raw_clean[TARGET].mean()
    total_pajak = df_raw_clean[TARGET].sum()
    eff_rate_avg = (df_raw_clean[TARGET] / df_raw_clean["penjualan_bersih"]).mean()

    print(f"\n  [3.1] INSIGHT KEUANGAN UTAMA:")
    print(f"        Rata-rata Beban Pajak/Kuartal  : Rp {avg_pajak:>18,.0f}")
    print(f"        Total Beban Pajak Historis     : Rp {total_pajak:>18,.0f}")
    print(f"        Effective Tax Rate (avg)       : {eff_rate_avg:.2%}")
    print(
        f"        Kuartal Pajak Tertinggi        : "
        f"{df_raw_clean.loc[df_raw_clean[TARGET].idxmax(), 'periode'] if 'periode' in df_raw_clean.columns else 'N/A'}"
    )
    print(
        f"        Kuartal Pajak Terendah         : "
        f"{df_raw_clean.loc[df_raw_clean[TARGET].idxmin(), 'periode'] if 'periode' in df_raw_clean.columns else 'N/A'}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3.2 — Visualisasi 4-Panel
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "EDA — Analisis Data Keuangan & Perpajakan Perusahaan (2015–2023)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # ── Panel 1 (kiri atas): Heatmap Korelasi Antar Variabel ────────────────
    corr_cols = [
        "penjualan_bersih",
        "hpp",
        "beban_operasional",
        "ppn_keluaran",
        "pph_badan",
        "total_beban_pajak",
    ]
    corr_labels = ["Penjualan", "HPP", "Bbn.Ops", "PPN", "PPh", "Total Pajak"]
    corr_matrix = df_raw_clean[corr_cols].select_dtypes(include=[float, int]).corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=corr_labels,
        yticklabels=corr_labels,
        ax=axes[0, 0],
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    axes[0, 0].set_title("① Korelasi Antar Variabel Keuangan", fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=30, labelsize=9)
    axes[0, 0].tick_params(axis="y", rotation=0, labelsize=9)

    # ── Definisi variabel sumbu-X (dipakai oleh Panel 2, 3, dan scatter) ────
    x_pos = range(len(df_raw_clean))
    x_labels = (
        df_raw_clean["periode"].tolist()
        if "periode" in df_raw_clean.columns
        else [str(i) for i in x_pos]
    )
    step = max(1, len(x_labels) // 9)

    # ── Panel 2 (kanan atas): Tren Beban Pajak per Kuartal ──────────────────
    if "periode" in df_raw_clean.columns:
        axes[0, 1].plot(
            x_pos,
            df_raw_clean[TARGET] / 1e9,
            "o-",
            color="#E74C3C",
            lw=2,
            ms=5,
            label="Total Pajak",
        )
        axes[0, 1].plot(
            x_pos,
            df_raw_clean["ppn_keluaran"] / 1e9,
            "s--",
            color="#3498DB",
            lw=1.5,
            ms=4,
            label="PPN (11%)",
        )
        axes[0, 1].plot(
            x_pos,
            df_raw_clean["pph_badan"] / 1e9,
            "^:",
            color="#2ECC71",
            lw=1.5,
            ms=4,
            label="PPh (22%)",
        )
        axes[0, 1].set_xticks(list(x_pos)[::step])
        axes[0, 1].set_xticklabels(
            x_labels[::step], rotation=45, ha="right", fontsize=8
        )
        axes[0, 1].set_ylabel("Miliar Rupiah (Rp)")
        axes[0, 1].set_title(
            "② Tren Historis Beban Pajak per Kuartal", fontweight="bold"
        )
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M")
        )

    # ── Panel 3 (kiri bawah): Stacked Area — Komposisi Penjualan ────────────
    if "periode" in df_raw_clean.columns:
        laba_bersih = (
            df_raw_clean["penjualan_bersih"]
            - df_raw_clean["hpp"]
            - df_raw_clean["beban_operasional"]
        ).clip(lower=0)  # pandas 2.x: gunakan lower= bukan min=

        axes[1, 0].stackplot(
            x_pos,
            df_raw_clean["hpp"] / 1e9,
            df_raw_clean["beban_operasional"] / 1e9,
            laba_bersih / 1e9,
            labels=["HPP (COGS)", "Beban Operasional", "Laba Sebelum Pajak"],
            colors=["#E74C3C", "#F39C12", "#27AE60"],
            alpha=0.82,
        )
        axes[1, 0].set_xticks(list(x_pos)[::step])
        axes[1, 0].set_xticklabels(
            x_labels[::step], rotation=45, ha="right", fontsize=8
        )
        axes[1, 0].set_ylabel("Miliar Rupiah (Rp)")
        axes[1, 0].set_title(
            "③ Komposisi Penjualan: HPP · Ops · Laba", fontweight="bold"
        )
        axes[1, 0].legend(loc="upper left", fontsize=8)

    # ── Panel 4 (kanan bawah): Boxplot — Distribusi Pajak per Kuartal ───────
    if "kuartal" in df_raw_clean.columns:
        kuartal_labels = {
            1: "Q1 (Jan–Mar)",
            2: "Q2 (Apr–Jun)",
            3: "Q3 (Jul–Sep)",
            4: "Q4 (Okt–Des)",
        }
        df_box = df_raw_clean.copy()
        df_box["kuartal_label"] = df_box["kuartal"].map(
            lambda x: kuartal_labels.get(int(x), str(x))
        )
        df_box["pajak_miliar"] = df_box[TARGET] / 1e9

        sns.boxplot(
            data=df_box,
            x="kuartal_label",
            y="pajak_miliar",
            palette=["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"],
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("④ Distribusi Beban Pajak per Kuartal", fontweight="bold")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Total Beban Pajak (Miliar Rp)")
        axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    if save_plots:
        path1 = os.path.join(OUTPUT_DIR, "eda_analysis.png")
        plt.savefig(path1, dpi=150, bbox_inches="tight")
        print(f"\n  ✔ Plot 4-panel EDA disimpan → {path1}")
    plt.show()
    plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # LANGKAH 3.3 — Scatter: Penjualan Bersih vs Total Beban Pajak
    # Memvisualisasikan kekuatan hubungan linear antara dua variabel utama.
    # ─────────────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 5))

    _q_scatter = (
        np.array(df_raw_clean["kuartal"].values, dtype=float)
        if "kuartal" in df_raw_clean.columns
        else np.ones(len(df_raw_clean))
    )
    sc = ax.scatter(
        np.array(df_raw_clean["penjualan_bersih"].values, dtype=float) / 1e9,
        np.array(df_raw_clean[TARGET].values, dtype=float) / 1e9,
        c=_q_scatter,
        cmap="viridis",
        s=90,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax, label="Kuartal (1=Q1, 2=Q2, 3=Q3, 4=Q4)")

    # Garis regresi sederhana untuk menunjukkan tren linier
    x_vals = np.array(df_raw_clean["penjualan_bersih"].values, dtype=float) / 1e9
    y_vals = np.array(df_raw_clean[TARGET].values, dtype=float) / 1e9
    m, b = np.polyfit(x_vals, y_vals, 1)
    ax.plot(
        sorted(x_vals),
        [m * xv + b for xv in sorted(x_vals)],
        "r--",
        alpha=0.55,
        lw=2,
        label=f"Regresi: y = {m:.3f}x + {b:.3f}",
    )

    ax.set_xlabel("Penjualan Bersih (Miliar Rp)", fontsize=11)
    ax.set_ylabel("Total Beban Pajak (Miliar Rp)", fontsize=11)
    ax.set_title(
        "Hubungan Penjualan Bersih vs Total Beban Pajak\n(warna = kuartal)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.2f}M"))

    plt.tight_layout()
    if save_plots:
        path2 = os.path.join(OUTPUT_DIR, "eda_scatter_penjualan.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"  ✔ Plot scatter disimpan    → {path2}\n")
    plt.show()
    plt.close()


# =============================================================================
# TAHAP 4 — PEMODELAN (MODELING)
# Melatih model Linear Regression untuk memprediksi total beban pajak.
# =============================================================================


def train_model(df_scaled: pd.DataFrame, scaler: StandardScaler) -> tuple:
    """
    Melatih model Linear Regression pada fitur yang sudah ternormalisasi.

    Mengapa Linear Regression?
      ✓ Target berupa nilai numerik kontinu → cocok untuk regresi
      ✓ Terdapat hubungan linear kuat antara Penjualan/HPP/Ops dan Pajak
      ✓ Koefisien mudah diinterpretasi oleh tim Finance & Akunting
      ✓ Komputasi ringan, baseline solid sebelum model lebih kompleks

    Catatan arsitektur:
      • shuffle=False pada train_test_split → WAJIB untuk data time-series
        agar urutan waktu (2015→2023) terjaga dan tidak bocor
      • Scaler di-fit ulang HANYA pada X_train untuk menghindari data leakage

    Args:
        df_scaled (pd.DataFrame)  : Data bersih dengan fitur ternormalisasi.
        scaler    (StandardScaler): Scaler yang sudah di-fit pada seluruh data bersih.

    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, feature_names)
    """
    print("=" * 70)
    print("  TAHAP 4 — PEMODELAN (LINEAR REGRESSION)")
    print("=" * 70)

    # ── Siapkan matriks fitur (X) dan vektor target (y) ──────────────────────
    # X menggunakan nilai SCALED; y tetap menggunakan nilai ASLI (Rupiah)
    # agar hasil prediksi dapat langsung diinterpretasi dalam Rupiah.
    X = np.array(df_scaled[FEATURES].values, dtype=float)  # Shape: (n_samples, 3)
    y = np.array(df_scaled[TARGET].values, dtype=float)  # Shape: (n_samples,)

    print(f"  Fitur (X)  : {FEATURES}")
    print(f"  Target (y) : {TARGET}")
    print(f"  Dimensi X  : {X.shape}  |  Dimensi y : {y.shape}")

    # ── Train-Test Split: 80% latih / 20% uji ────────────────────────────────
    # shuffle=False: Data tetap urut berdasarkan waktu (kronologis)
    # Ini mensimulasikan skenario nyata di mana kita melatih model pada
    # data masa lalu dan mengujinya pada data yang lebih baru.
    _split = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, shuffle=False
    )
    X_train: np.ndarray = np.array(_split[0])
    X_test: np.ndarray = np.array(_split[1])
    y_train: np.ndarray = np.array(_split[2])
    y_test: np.ndarray = np.array(_split[3])

    print(
        f"\n  Train set  : {X_train.shape[0]} sampel ({X_train.shape[0] / len(X) * 100:.0f}%)"
    )
    print(
        f"  Test set   : {X_test.shape[0]} sampel ({X_test.shape[0] / len(X) * 100:.0f}%)"
    )

    # ── Latih Model Linear Regression ────────────────────────────────────────
    # fit_intercept=True: Model akan mencari nilai intercept (konstanta dasar pajak)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # ── Tampilkan Koefisien — Interpretasi untuk Tim Finance ─────────────────
    print("\n  Persamaan Regresi Linear yang Ditemukan Model:")
    print(f"  ┌────────────────────────────────────────────────┐")
    print(f"  │  Pajak_Prediksi = {model.intercept_:>+.2f} (intercept)    │")
    for feat, coef in zip(FEATURES, model.coef_):
        arah = "↑ naik" if coef > 0 else "↓ turun"
        print(f"  │    + ({coef:>+.4f}) × {feat:<22} [{arah}]  │")
    print(f"  └────────────────────────────────────────────────┘")

    print("\n  Interpretasi Koefisien (unit: scaled → 1σ perubahan fitur):")
    print("  Koefisien positif = kenaikan fitur → prediksi pajak NAIK")
    print("  Koefisien negatif = kenaikan fitur → prediksi pajak TURUN\n")

    return model, X_train, X_test, y_train, y_test, FEATURES


# =============================================================================
# TAHAP 5 — EVALUASI MODEL
# Mengukur akurasi prediksi model menggunakan metrik standar regresi.
# =============================================================================


def evaluate_model(
    model: LinearRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_scaled: pd.DataFrame,
    save_plots: bool = True,
) -> dict:
    """
    Mengevaluasi performa model dan mencetak laporan lengkap.

    Metrik yang digunakan:
      • MAE  — Mean Absolute Error    : Rata-rata kesalahan absolut prediksi (Rp)
                                        Mudah dipahami tim Finance sebagai "rata-rata meleset sekian Rp"
      • RMSE — Root Mean Squared Error: Akar MSE, memberi penalti lebih besar pada error besar
                                        Lebih sensitif terhadap prediksi yang sangat meleset
      • R²   — R-Squared              : 1.0 = model sempurna, 0.0 = sama dengan prediksi rata-rata
                                        Menunjukkan persentase variansi data yang bisa dijelaskan model
      • MAPE — Mean Absolute Pct Error: Error dalam persentase, mudah dibandingkan antar skala dataset

    Args:
        model     : LinearRegression yang sudah dilatih.
        X_test    : Matriks fitur data uji (scaled).
        y_test    : Vektor target aktual data uji (nilai Rupiah asli).
        df_scaled : DataFrame lengkap untuk plot prediksi penuh sepanjang timeline.
        save_plots: Jika True, simpan plot evaluasi ke OUTPUT_DIR.

    Returns:
        dict: {"MAE": float, "MSE": float, "RMSE": float, "R2": float, "MAPE": float}
    """
    print("=" * 70)
    print("  TAHAP 5 — EVALUASI MODEL")
    print("=" * 70)

    # ── Prediksi pada data uji ─────────────────────────────────────────────
    y_pred = model.predict(X_test)

    # ── Hitung semua metrik evaluasi ───────────────────────────────────────
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # MAPE: hindari pembagian nol jika y_test ada yang bernilai 0
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

    # ── Cetak laporan evaluasi terstruktur ─────────────────────────────────
    print(f"\n  ┌{'─' * 52}┐")
    print(f"  │{'  LAPORAN EVALUASI MODEL  ':^52}│")
    print(f"  ├{'─' * 52}┤")
    print(f"  │  MAE   (Mean Absolute Error)   : Rp {mae:>15,.0f}  │")
    print(f"  │  RMSE  (Root Mean Sq. Error)   : Rp {rmse:>15,.0f}  │")
    print(f"  │  R²    (R-Squared)             :    {r2:>15.4f}  │")
    print(f"  │  MAPE  (Mean Abs. Pct Error)   :    {mape:>14.2f}%  │")
    print(f"  └{'─' * 52}┘")

    # Penilaian otomatis berdasarkan nilai R²
    if r2 >= 0.90:
        grade = "🟢 SANGAT BAIK — Model siap digunakan untuk estimasi produksi"
    elif r2 >= 0.75:
        grade = "🟡 BAIK — Validasi tambahan disarankan sebelum ke produksi"
    elif r2 >= 0.50:
        grade = "🟠 CUKUP — Pertimbangkan feature engineering lebih lanjut"
    else:
        grade = "🔴 KURANG — Coba Gradient Boosting / Random Forest / Polynomial"
    print(f"\n  Penilaian : {grade}\n")

    # ── Visualisasi Evaluasi ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Evaluasi Model Linear Regression — Prediksi vs Aktual Beban Pajak",
        fontweight="bold",
    )

    # Panel kiri: Scatter aktual vs prediksi pada data uji
    ax1 = axes[0]
    ax1.scatter(
        y_test / 1e9,
        y_pred / 1e9,
        color="#E74C3C",
        s=80,
        edgecolors="white",
        zorder=5,
        label="Data Uji",
    )
    # Garis diagonal sempurna (jika prediksi = aktual, semua titik ada di garis ini)
    min_lim = min(y_test.min(), y_pred.min()) / 1e9 * 0.95
    max_lim = max(y_test.max(), y_pred.max()) / 1e9 * 1.05
    ax1.plot(
        [min_lim, max_lim],
        [min_lim, max_lim],
        "k--",
        alpha=0.5,
        label="Prediksi Sempurna (y=x)",
    )
    ax1.set_xlim(min_lim, max_lim)
    ax1.set_ylim(min_lim, max_lim)
    ax1.set_xlabel("Aktual (Miliar Rp)")
    ax1.set_ylabel("Prediksi (Miliar Rp)")
    ax1.set_title(
        f"Scatter: Aktual vs Prediksi\nR² = {r2:.4f}  |  MAPE = {mape:.2f}%",
        fontweight="bold",
    )
    ax1.legend(fontsize=9)

    # Panel kanan: Timeline prediksi vs aktual sepanjang seluruh dataset
    ax2 = axes[1]
    y_all = np.array(df_scaled[TARGET].values, dtype=float)
    y_hat = np.array(
        model.predict(np.array(df_scaled[FEATURES].values, dtype=float)), dtype=float
    )
    x_idx = range(len(df_scaled))

    ax2.plot(x_idx, y_all / 1e9, "o-", color="#2C3E50", lw=2, ms=5, label="Aktual")
    ax2.plot(x_idx, y_hat / 1e9, "s--", color="#E74C3C", lw=2, ms=4, label="Prediksi")

    # Garis vertikal pemisah train/test
    n_train = len(df_scaled) - len(y_test)
    ax2.axvline(x=n_train - 0.5, color="gray", linestyle=":", lw=1.5, alpha=0.7)
    ax2.text(
        n_train - 0.3,
        y_all.min() / 1e9,
        "← Train | Test →",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    # Label sumbu X: label kuartal
    if "periode" in df_scaled.columns:
        step_x = max(1, len(df_scaled) // 8)
        ax2.set_xticks(list(x_idx)[::step_x])
        ax2.set_xticklabels(
            df_scaled["periode"].iloc[::step_x].tolist(),
            rotation=45,
            ha="right",
            fontsize=7,
        )
    ax2.set_ylabel("Beban Pajak (Miliar Rp)")
    ax2.set_title("Timeline: Prediksi Model vs Data Aktual", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))

    plt.tight_layout()
    if save_plots:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, "evaluasi_model.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✔ Plot evaluasi disimpan   → {path}\n")
    plt.show()
    plt.close()

    return metrics


# =============================================================================
# TAHAP 6 — SIMPAN & MUAT ARTEFAK MODEL
# Menyimpan semua komponen pipeline ke satu file pickle agar dapat
# dimuat kembali oleh dashboard Streamlit tanpa melatih ulang.
# =============================================================================


def save_artefak(
    model: LinearRegression,
    scaler: StandardScaler,
    df_raw_clean: pd.DataFrame,
    df_scaled: pd.DataFrame,
    metrics: dict,
) -> None:
    """
    Menyimpan semua artefak pipeline ke file pickle (model_artefak.pkl).

    Komponen yang disimpan:
      • model        → LinearRegression yang sudah dilatih
      • scaler       → StandardScaler yang sudah di-fit (untuk transform input baru)
      • df_raw_clean → Data bersih pra-scaling (untuk tabel & EDA di dashboard)
      • df_scaled    → Data bersih pasca-scaling (untuk prediksi & grafik model)
      • metrics      → Kamus metrik evaluasi (MAE, RMSE, R², MAPE)
      • features     → Daftar nama kolom fitur
      • target       → Nama kolom target
      • ppn_rate     → Tarif PPN saat ini
      • pph_rate     → Tarif PPh saat ini
    """
    artefak = {
        "model": model,
        "scaler": scaler,
        "df_raw_clean": df_raw_clean,
        "df_scaled": df_scaled,
        "metrics": metrics,
        "features": FEATURES,
        "target": TARGET,
        "ppn_rate": TAX_PPN_RATE,
        "pph_rate": TAX_PPH_RATE,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artefak, f)

    print("=" * 70)
    print("  TAHAP 6 — SIMPAN ARTEFAK MODEL")
    print("=" * 70)
    print(f"  ✔ Artefak pipeline tersimpan di : '{MODEL_PATH}'")
    print(f"  ✔ Komponen: model, scaler, data, metrics, rates")
    print(f"  ✔ Jalankan dashboard            : streamlit run app_dashboard.py\n")


def load_artefak() -> dict:
    """
    Memuat artefak model dari file pickle.

    Returns:
        dict: Kamus berisi semua artefak pipeline.

    Raises:
        FileNotFoundError: Jika file artefak belum tersedia.
                           Solusi: jalankan 'python tax_pipeline.py' terlebih dahulu.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"File artefak '{MODEL_PATH}' tidak ditemukan!\n"
            f"Jalankan terlebih dahulu: python tax_pipeline.py"
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# =============================================================================
# FUNGSI PREDIKSI — KUARTAL MENDATANG
# Menggunakan model dan scaler yang sudah ada untuk memprediksi beban pajak.
# =============================================================================


def predict_next_quarter(
    model: LinearRegression,
    scaler: StandardScaler,
    penjualan_bersih: float,
    hpp: float,
    beban_operasional: float,
) -> dict:
    """
    Memprediksi total beban pajak untuk kuartal mendatang berdasarkan estimasi input.

    Alur Prediksi:
      1. Susun array input 2D → shape (1, 3)
      2. Scale input dengan scaler.transform() — JANGAN fit_transform() lagi!
         (re-fitting scaler pada data baru akan menggeser μ dan σ → prediksi salah)
      3. Jalankan model.predict() → dapatkan total_beban_pajak
      4. Hitung komponen PPh & PPN secara rule-based untuk breakdown

    Args:
        model             : LinearRegression yang sudah dilatih.
        scaler            : StandardScaler yang sudah di-fit pada data training.
        penjualan_bersih  : Estimasi pendapatan bersih kuartal mendatang (Rp).
        hpp               : Estimasi harga pokok penjualan (Rp).
        beban_operasional : Estimasi beban operasional (Rp).

    Returns:
        dict: {
          "total_beban_pajak"  : Prediksi total pajak dari model (Rp),
          "ppn_keluaran"       : PPN = 11% × penjualan (rule-based) (Rp),
          "pph_badan"          : PPh = 22% × laba kena pajak (rule-based) (Rp),
          "laba_kena_pajak"    : Laba sebelum pajak yang positif (Rp),
          "effective_tax_rate" : Total pajak / penjualan × 100 (%)
        }
    """
    # Susun array input: shape (1, n_features) yang dibutuhkan sklearn
    X_input = np.array([[penjualan_bersih, hpp, beban_operasional]])

    # Scale fitur menggunakan μ dan σ dari data training (transform only!)
    X_scaled = scaler.transform(X_input)

    # Jalankan prediksi model → total beban pajak dalam Rupiah
    total_pred = float(model.predict(X_scaled)[0])
    total_pred = max(0.0, total_pred)  # Beban pajak tidak boleh negatif

    # Kalkulasi komponen pajak secara rule-based (sesuai regulasi UU HPP 2022)
    ppn_pred = penjualan_bersih * TAX_PPN_RATE
    laba_kena = max(0.0, penjualan_bersih - hpp - beban_operasional)
    pph_pred = laba_kena * TAX_PPH_RATE
    eff_rate = (total_pred / penjualan_bersih * 100) if penjualan_bersih > 0 else 0.0

    return {
        "total_beban_pajak": total_pred,
        "ppn_keluaran": ppn_pred,
        "pph_badan": pph_pred,
        "laba_kena_pajak": laba_kena,
        "effective_tax_rate": eff_rate,
    }


# =============================================================================
# MAIN — ORKESTRATOR PIPELINE LENGKAP
# Menjalankan seluruh 6 tahap secara berurutan.
# =============================================================================


def main() -> None:
    """
    Mengorkestrasikan seluruh pipeline ML dari awal hingga akhir.

    Urutan eksekusi:
      1. generate_erp_data()       → Data Ingestion
      2. preprocess_data()         → Cleaning + Normalisasi
      3. run_eda()                 → EDA + Visualisasi
      4. train_model()             → Linear Regression Training
      5. evaluate_model()          → MAE, RMSE, R², MAPE
      6. save_artefak()            → Pickle semua komponen
      7. predict_next_quarter()    → Demo prediksi kuartal 2024-Q1
    """
    print()
    print("=" * 70)
    print("  🏦  SISTEM PREDIKSI BEBAN PAJAK PPh / PPN — KUARTAL MENDATANG")
    print("  📋  Regulasi: UU HPP No.7 Tahun 2021 (PPh 22% | PPN 11%)")
    print("=" * 70)
    print()

    # ── Tahap 1: Data Ingestion ───────────────────────────────────────────────
    df_raw = generate_erp_data()

    # ── Tahap 2: Preprocessing ───────────────────────────────────────────────
    df_scaled, scaler, df_raw_clean = preprocess_data(df_raw)

    # ── Tahap 3: EDA ─────────────────────────────────────────────────────────
    run_eda(df_raw_clean, save_plots=True)

    # ── Tahap 4: Modeling ────────────────────────────────────────────────────
    model, X_train, X_test, y_train, y_test, feature_names = train_model(
        df_scaled, scaler
    )

    # ── Tahap 5: Evaluasi ────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test, df_scaled, save_plots=True)

    # ── Tahap 6: Simpan Artefak ──────────────────────────────────────────────
    save_artefak(model, scaler, df_raw_clean, df_scaled, metrics)

    # ── DEMO: Prediksi Beban Pajak 2024-Q1 ───────────────────────────────────
    print("=" * 70)
    print("  DEMO — PREDIKSI BEBAN PAJAK KUARTAL MENDATANG (2024-Q1)")
    print("=" * 70)

    hasil = predict_next_quarter(
        model=model,
        scaler=scaler,
        penjualan_bersih=9_200_000_000,  # Estimasi Rp 9,2 Miliar
        hpp=5_300_000_000,  # Estimasi HPP Rp 5,3 Miliar
        beban_operasional=1_400_000_000,  # Estimasi Ops Rp 1,4 Miliar
    )

    print(f"\n  Input Estimasi Keuangan 2024-Q1:")
    print(f"    • Penjualan Bersih       : Rp  9.200.000.000")
    print(f"    • HPP / COGS             : Rp  5.300.000.000")
    print(f"    • Beban Operasional      : Rp  1.400.000.000")
    print(f"    • Laba Kena Pajak        : Rp {hasil['laba_kena_pajak']:>15,.0f}")
    print(f"\n  Hasil Prediksi Model:")
    print(f"    ┌──────────────────────────────────────────────────┐")
    print(f"    │  PPN Keluaran (11%)    : Rp {hasil['ppn_keluaran']:>15,.0f}  │")
    print(f"    │  PPh Badan (22%)       : Rp {hasil['pph_badan']:>15,.0f}  │")
    print(f"    │  ─────────────────────────────────────────────   │")
    print(f"    │  TOTAL BEBAN PAJAK     : Rp {hasil['total_beban_pajak']:>15,.0f}  │")
    print(f"    │  Effective Tax Rate    :    {hasil['effective_tax_rate']:>14.2f}%  │")
    print(f"    └──────────────────────────────────────────────────┘")
    print(f"\n  ✅ Pipeline selesai. Jalankan: streamlit run app_dashboard.py\n")


# Entry point
if __name__ == "__main__":
    main()