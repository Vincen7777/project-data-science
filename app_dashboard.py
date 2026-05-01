# -*- coding: utf-8 -*-
"""
=============================================================================
  DASHBOARD STREAMLIT — PREDIKSI ESTIMASI BEBAN PAJAK PPh/PPN
  KUARTAL MENDATANG
=============================================================================
  Dashboard interaktif untuk memvisualisasikan data historis, hasil EDA,
  performa model, dan prediksi beban pajak berdasarkan estimasi input
  tim Finance.

  Cara menjalankan:
    streamlit run app_dashboard.py

  Prasyarat:
    Pastikan tax_pipeline.py telah dijalankan setidaknya sekali:
    → python tax_pipeline.py

  Jika artefak (model_artefak.pkl) belum ada, dashboard akan melatih
  model secara otomatis di latar belakang saat pertama kali dibuka.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORT LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import warnings

import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Import fungsi dan konstanta dari pipeline utama
# Ini memastikan satu sumber kebenaran (single source of truth) untuk
# semua logika ML dan konstanta perpajakan.
from sklearn.linear_model import LinearRegression

from tax_pipeline import (
    FEATURES,
    MODEL_PATH,
    REQUIRED_INPUT_COLS,
    TARGET,
    TAX_PPH_RATE,
    TAX_PPN_RATE,
    USER_DATA_PATH,
    buat_template_csv,
    evaluate_model,
    generate_erp_data,
    hitung_kolom_pajak,
    load_artefak,
    load_user_data,
    predict_next_quarter,
    preprocess_data,
    save_artefak,
    save_user_data,
    train_model,
    validasi_input_data,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  KONFIGURASI HALAMAN STREAMLIT
# Harus dipanggil PERTAMA sebelum perintah st lainnya.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io",
        "About": "🏦 Dashboard Prediksi Beban Pajak PPh/PPN — UU HPP 2022",
    },
)

# ─── Injeksi CSS Kustom untuk mempercantik tampilan ─────────────────────────
# Streamlit mendukung HTML/CSS melalui st.markdown dengan unsafe_allow_html=True.
st.markdown(
    """
    <style>
    /* Kartu metrik kustom dengan gradien gelap */
    .kpi-card {
        background  : linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding     : 18px 22px;
        border-left : 5px solid #e74c3c;
        margin      : 6px 0;
    }
    .kpi-title { color: #bdc3c7; font-size: 12px; letter-spacing: 0.8px; }
    .kpi-value { color: #ecf0f1; font-size: 20px; font-weight: 700; margin-top: 4px; }

    /* Header seksi berwarna gradien */
    .section-header {
        background   : linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        color        : white;
        padding      : 9px 18px;
        border-radius: 8px;
        font-weight  : bold;
        font-size    : 14px;
        margin       : 18px 0 10px 0;
    }

    /* Kotak info biru */
    .info-box {
        background  : #eaf4fb;
        border-left : 4px solid #3498db;
        padding     : 11px 15px;
        border-radius: 6px;
        margin      : 8px 0;
        font-size   : 13px;
    }

    /* Kotak peringatan kuning */
    .warn-box {
        background  : #fef9e7;
        border-left : 4px solid #f39c12;
        padding     : 11px 15px;
        border-radius: 6px;
        margin      : 8px 0;
        font-size   : 13px;
    }

    /* Kotak sukses hijau */
    .success-box {
        background  : #eafaf1;
        border-left : 4px solid #2ecc71;
        padding     : 11px 15px;
        border-radius: 6px;
        margin      : 8px 0;
        font-size   : 13px;
    }

    /* Banner prediksi merah besar */
    .prediction-banner {
        background    : linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color         : white;
        padding       : 24px 30px;
        border-radius : 14px;
        text-align    : center;
        margin        : 18px 0;
        box-shadow    : 0 6px 20px rgba(231,76,60,0.30);
    }
    .prediction-banner h3 { margin: 0 0 8px 0; color: white; font-size: 16px; opacity: 0.92; }
    .prediction-banner h1 { margin: 0 0 10px 0; color: white; font-size: 38px; }
    .prediction-banner p  { margin: 0; opacity: 0.85; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FUNGSI PEMBANTU (HELPER FUNCTIONS)
# ─────────────────────────────────────────────────────────────────────────────


def fmt_rp(nilai: float, singkat: bool = False) -> str:
    """
    Memformat angka menjadi string Rupiah Indonesia.

    Args:
        nilai   : Nilai numerik dalam Rupiah penuh.
        singkat : Jika True, tampilkan dalam format singkat (M=Miliar, Jt=Juta).

    Returns:
        str: String terformat, misal "Rp 1.25M" atau "Rp 1.250.000.000"
    """
    if singkat:
        if abs(nilai) >= 1_000_000_000:
            return f"Rp {nilai / 1e9:.2f}M"  # Miliar
        elif abs(nilai) >= 1_000_000:
            return f"Rp {nilai / 1e6:.1f}Jt"  # Juta
        else:
            return f"Rp {nilai:,.0f}"
    return f"Rp {nilai:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FUNGSI CACHE: MUAT ATAU LATIH MODEL
# @st.cache_resource memastikan pipeline hanya dieksekusi SEKALI per sesi
# server (tidak diulang setiap kali widget berinteraksi).
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="⏳ Memuat model pipeline...")
def get_artefak() -> dict:
    """
    Memuat artefak model dari pickle. Jika belum ada, latih model baru otomatis.

    Menggunakan st.cache_resource agar:
      ✓ Model hanya dilatih sekali, bukan setiap interaksi widget
      ✓ Menghemat waktu komputasi secara signifikan
      ✓ Aman untuk multi-user (setiap session berbagi resource yang sama)

    ⚠️  ATURAN PENTING @st.cache_resource:
        JANGAN letakkan perintah st.* (st.toast, st.write, st.warning, dll.)
        di dalam fungsi ini. Streamlit tidak bisa me-replay perintah UI
        saat cache diambil dari memory → CacheReplayClosureError.
        Semua notifikasi UI harus diletakkan di luar fungsi ini (di main()).

    Returns:
        dict: Kamus berisi model, scaler, data, metrics, dan konstanta.
    """
    if os.path.exists(MODEL_PATH):
        # File artefak sudah ada → muat langsung tanpa perintah UI apapun
        return load_artefak()
    else:
        # Artefak belum tersedia → jalankan pipeline otomatis.
        # Prioritas data: (1) data pengguna → (2) data simulasi ERP
        # Tidak ada st.* di sini! Notifikasi ditangani di main().
        df_user = load_user_data()
        df_raw = df_user if df_user is not None else generate_erp_data()
        df_scaled, scaler, df_raw_clean = preprocess_data(df_raw)
        model, _, X_test, _, y_test, _ = train_model(df_scaled, scaler)
        metrics = evaluate_model(model, X_test, y_test, df_scaled, save_plots=False)
        save_artefak(model, scaler, df_raw_clean, df_scaled, metrics)
        return load_artefak()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FUNGSI PLOT (TERPISAH DARI UI AGAR MUDAH DIUJI & DIRAWAT)
# ─────────────────────────────────────────────────────────────────────────────


def plot_tren_pajak(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Membuat line chart tren historis PPN, PPh, dan Total Pajak per kuartal.

    Args:
        df: DataFrame bersih dengan kolom periode, ppn_keluaran, pph_badan, total_beban_pajak.

    Returns:
        plt.Figure: Objek figure matplotlib siap ditampilkan di Streamlit.
    """
    fig, ax = plt.subplots(figsize=(11, 4))
    x = range(len(df))

    ax.plot(
        x,
        df[TARGET] / 1e9,
        "o-",
        color="#E74C3C",
        lw=2.5,
        ms=6,
        label="Total Beban Pajak",
    )
    ax.plot(
        x,
        df["ppn_keluaran"] / 1e9,
        "s--",
        color="#3498DB",
        lw=1.8,
        ms=5,
        label="PPN Keluaran (11%)",
    )
    ax.plot(
        x,
        df["pph_badan"] / 1e9,
        "^:",
        color="#2ECC71",
        lw=1.8,
        ms=5,
        label="PPh Badan (22%)",
    )

    # Arsir area antara PPN dan Total untuk visualisasi kontribusi PPh
    ax.fill_between(
        x,
        df["ppn_keluaran"] / 1e9,
        df[TARGET] / 1e9,
        alpha=0.12,
        color="#2ECC71",
        label="Kontribusi PPh (area)",
    )

    # Konfigurasi label sumbu X
    step = max(1, len(df) // 10)
    labels = df["periode"].tolist() if "periode" in df.columns else [str(i) for i in x]
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)

    ax.set_ylabel("Miliar Rupiah (Rp)", fontsize=10)
    ax.set_title(
        "Tren Historis Beban Pajak per Kuartal (2015–2023)",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    return fig


def plot_heatmap_korelasi(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Membuat heatmap korelasi antar variabel keuangan utama.

    Args:
        df: DataFrame bersih dengan semua kolom keuangan.

    Returns:
        plt.Figure: Heatmap korelasi dengan anotasi nilai.
    """
    cols = [
        "penjualan_bersih",
        "hpp",
        "beban_operasional",
        "ppn_keluaran",
        "pph_badan",
        "total_beban_pajak",
    ]
    labels = ["Penjualan", "HPP", "Bbn.Ops", "PPN", "PPh", "Total Pajak"]
    corr = df[cols].select_dtypes(include=[float, int]).corr()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.78},
        annot_kws={"size": 9, "weight": "bold"},
    )
    ax.set_title(
        "Heatmap Korelasi Antar Variabel Keuangan",
        fontweight="bold",
        pad=12,
        fontsize=11,
    )
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    return fig


def plot_boxplot_kuartal(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Membuat boxplot distribusi beban pajak per kuartal.

    Args:
        df: DataFrame bersih dengan kolom 'kuartal' dan 'total_beban_pajak'.

    Returns:
        plt.Figure: Boxplot per kuartal (Q1–Q4).
    """
    kuartal_map = {
        1: "Q1 (Jan–Mar)",
        2: "Q2 (Apr–Jun)",
        3: "Q3 (Jul–Sep)",
        4: "Q4 (Okt–Des)",
    }
    df_box = df.copy()
    df_box["kuartal_label"] = df_box["kuartal"].map(
        lambda x: kuartal_map.get(int(x), str(x))
    )
    df_box["pajak_miliar"] = df_box[TARGET] / 1e9

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df_box,
        x="kuartal_label",
        y="pajak_miliar",
        palette=["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"],
        ax=ax,
        width=0.55,
    )
    ax.set_title("Distribusi Beban Pajak per Kuartal", fontweight="bold", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Total Beban Pajak (Miliar Rp)", fontsize=10)
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_aktual_vs_prediksi(
    model: "LinearRegression",
    df_scaled: pd.DataFrame,
    n_test: int,
) -> matplotlib.figure.Figure:
    """
    Membuat dua-panel plot: timeline aktual vs prediksi + scatter aktual vs prediksi.

    Args:
        model    : Model LinearRegression yang sudah dilatih.
        df_scaled: DataFrame dengan fitur ternormalisasi dan target asli.
        n_test   : Jumlah sampel test set.

    Returns:
        plt.Figure: Plot dua panel evaluasi model.
    """
    y_actual = np.array(df_scaled[TARGET].values, dtype=float)
    y_pred = np.array(
        model.predict(np.array(df_scaled[FEATURES].values, dtype=float)), dtype=float
    )
    x_idx = range(len(df_scaled))
    n_train = len(df_scaled) - n_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # ── Panel kiri: Timeline aktual vs prediksi ──────────────────────────────
    axes[0].plot(
        x_idx, y_actual / 1e9, "o-", color="#2C3E50", lw=2.2, ms=5.5, label="Aktual"
    )
    axes[0].plot(
        x_idx,
        y_pred / 1e9,
        "s--",
        color="#E74C3C",
        lw=2.2,
        ms=5,
        label="Prediksi Model",
    )

    # Garis vertikal pemisah train / test
    axes[0].axvline(x=n_train - 0.5, color="#95a5a6", linestyle=":", lw=2)
    axes[0].text(
        n_train - 0.3,
        min(y_actual.min(), y_pred.min()) / 1e9 * 1.01,
        "← Train | Test →",
        fontsize=8,
        color="#7f8c8d",
        va="bottom",
    )

    # Label sumbu X — periode kuartal
    if "periode" in df_scaled.columns:
        step_x = max(1, len(df_scaled) // 9)
        axes[0].set_xticks(list(x_idx)[::step_x])
        axes[0].set_xticklabels(
            df_scaled["periode"].iloc[::step_x].tolist(),
            rotation=45,
            ha="right",
            fontsize=7,
        )
    axes[0].set_ylabel("Beban Pajak (Miliar Rp)", fontsize=10)
    axes[0].set_title("Timeline: Aktual vs Prediksi Model", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))

    # ── Panel kanan: Scatter aktual vs prediksi ──────────────────────────────
    axes[1].scatter(
        y_actual / 1e9,
        y_pred / 1e9,
        color="#E74C3C",
        s=70,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    mn = min(y_actual.min(), y_pred.min()) / 1e9 * 0.95
    mx = max(y_actual.max(), y_pred.max()) / 1e9 * 1.05
    axes[1].plot(
        [mn, mx], [mn, mx], "k--", alpha=0.4, lw=1.5, label="Prediksi Sempurna (y = x)"
    )
    axes[1].set_xlim(mn, mx)
    axes[1].set_ylim(mn, mx)
    axes[1].set_xlabel("Aktual (Miliar Rp)", fontsize=10)
    axes[1].set_ylabel("Prediksi (Miliar Rp)", fontsize=10)
    axes[1].set_title("Scatter: Aktual vs Prediksi", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.35)

    plt.tight_layout()
    return fig


def plot_pie_komposisi(
    ppn: float, pph: float, label: str = ""
) -> matplotlib.figure.Figure:
    """
    Membuat pie chart komposisi PPh vs PPN pada prediksi kuartal mendatang.

    Args:
        ppn  : Nilai PPN Keluaran (Rp).
        pph  : Nilai PPh Badan (Rp).
        label: Label judul tambahan (misal nama kuartal).

    Returns:
        plt.Figure: Pie chart dua irisan.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    total = ppn + pph
    if total <= 0:
        ax.text(
            0.5,
            0.5,
            "Tidak ada pajak\n(Perusahaan Merugi)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        plt.tight_layout()
        return fig

    sizes = [ppn, pph]
    labels = [
        f"PPN Keluaran\n{fmt_rp(ppn, singkat=True)}",
        f"PPh Badan\n{fmt_rp(pph, singkat=True)}",
    ]
    colors = ["#3498DB", "#2ECC71"]
    explode = (0.05, 0.05)

    pie_result = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=explode,
        textprops={"fontsize": 10},
    )
    autotexts = pie_result[2] if len(pie_result) > 2 else []
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_fontsize(11)

    ax.set_title(
        f"Komposisi Beban Pajak\n{label}",
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    plt.tight_layout()
    return fig


def plot_scatter_penjualan_pajak(df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Scatter plot: Penjualan Bersih vs Total Beban Pajak dengan garis regresi.

    Args:
        df: DataFrame bersih.

    Returns:
        plt.Figure: Scatter plot dengan garis trend linier.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    q_vals = (
        np.array(df["kuartal"].values, dtype=float)
        if "kuartal" in df.columns
        else np.ones(len(df))
    )
    sc = ax.scatter(
        np.array(df["penjualan_bersih"].values, dtype=float) / 1e9,
        np.array(df[TARGET].values, dtype=float) / 1e9,
        c=q_vals,
        cmap="viridis",
        s=85,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
    )
    plt.colorbar(sc, ax=ax, label="Kuartal (1=Q1 … 4=Q4)")

    # Tambah garis regresi linier sederhana
    x_v = np.array(df["penjualan_bersih"].values, dtype=float) / 1e9
    y_v = np.array(df[TARGET].values, dtype=float) / 1e9
    m, b = np.polyfit(x_v, y_v, 1)
    ax.plot(
        sorted(x_v),
        [m * xv + b for xv in sorted(x_v)],
        "r--",
        alpha=0.6,
        lw=2,
        label=f"Tren: y = {m:.3f}x + {b:.3f}",
    )

    ax.set_xlabel("Penjualan Bersih (Miliar Rp)", fontsize=11)
    ax.set_ylabel("Total Beban Pajak (Miliar Rp)", fontsize=11)
    ax.set_title(
        "Hubungan Penjualan Bersih vs Total Beban Pajak",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.0f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.2f}M"))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FUNGSI ANIMASI GRAFIK
#     Setiap fungsi memakai st.empty() sebagai placeholder yang di-overwrite
#     setiap frame sehingga menghasilkan efek grafik "berjalan".
# ─────────────────────────────────────────────────────────────────────────────


def animasi_tren_pajak(df: pd.DataFrame, speed: float = 0.07) -> None:
    """
    Animasi line-chart tren pajak: titik digambar satu per satu per kuartal.

    Cara kerja:
      1. Buat st.empty() sebagai kanvas
      2. Loop i=1..n: render df.iloc[:i] ke kanvas yang sama
      3. plt.close() setiap frame agar memori tidak bocor
      4. time.sleep(speed) untuk mengatur laju animasi

    Args:
        df    : DataFrame bersih (pra-scaling) dengan kolom pajak.
        speed : Jeda antar frame dalam detik (kecil = lebih cepat).
    """
    n = len(df)
    x_all = list(range(n))
    labels = (
        df["periode"].tolist() if "periode" in df.columns else [str(i) for i in x_all]
    )
    step = max(1, n // 10)  # Maks 10 label sumbu-X
    y_max = df[TARGET].max() / 1e9 * 1.20  # Batas atas tetap agar tidak loncat

    kanvas = st.empty()  # Placeholder grafik
    progress = st.progress(0, text="Memulai animasi...")

    for i in range(1, n + 1):
        df_part = df.iloc[:i]
        x_part = list(range(i))
        periode = (
            df_part["periode"].iloc[-1] if "periode" in df_part.columns else f"#{i}"
        )

        fig, ax = plt.subplots(figsize=(11, 4))
        sns.set_theme(style="whitegrid", palette="muted")

        # ── Gambar ketiga garis secara progresif ──────────────────────────
        ax.plot(
            x_part,
            df_part[TARGET] / 1e9,
            "o-",
            color="#E74C3C",
            lw=2.5,
            ms=6,
            label="Total Beban Pajak",
        )
        ax.plot(
            x_part,
            df_part["ppn_keluaran"] / 1e9,
            "s--",
            color="#3498DB",
            lw=1.8,
            ms=5,
            label="PPN (11%)",
        )
        ax.plot(
            x_part,
            df_part["pph_badan"] / 1e9,
            "^:",
            color="#2ECC71",
            lw=1.8,
            ms=5,
            label="PPh (22%)",
        )

        # ── Area kontribusi PPh (muncul setelah ≥ 2 titik) ───────────────
        if i > 1:
            ax.fill_between(
                x_part,
                df_part["ppn_keluaran"] / 1e9,
                df_part[TARGET] / 1e9,
                alpha=0.13,
                color="#2ECC71",
            )

        # ── Anotasi nilai terkini dengan callout box ──────────────────────
        val_terkini = df_part[TARGET].iloc[-1] / 1e9
        ax.annotate(
            f" {periode}\nRp{val_terkini:.2f}M",
            xy=(i - 1, val_terkini),
            xytext=(10, 14),
            textcoords="offset points",
            fontsize=8,
            color="#E74C3C",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.2),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#E74C3C",
                alpha=0.85,
            ),
        )

        # ── Sumbu & dekorasi ──────────────────────────────────────────────
        ax.set_xlim(-0.5, n - 0.5)  # Lebar tetap agar tidak melompat
        ax.set_ylim(0, y_max)
        ax.set_xticks(x_all[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Miliar Rupiah (Rp)", fontsize=10)
        ax.set_title(
            f"Tren Historis Beban Pajak — {periode}  [{i}/{n} kuartal]",
            fontweight="bold",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))
        ax.grid(axis="y", alpha=0.4)

        plt.tight_layout()
        kanvas.pyplot(fig, use_container_width=True)  # Overwrite kanvas yang sama
        plt.close(fig)  # Bebaskan memori setiap frame
        progress.progress(i / n, text=f"Kuartal {periode}  ({i}/{n})")
        time.sleep(speed)

    progress.empty()  # Hapus progress bar setelah selesai
    st.toast("✅ Animasi tren selesai!", icon="📈")


def animasi_scatter(df: pd.DataFrame, speed: float = 0.10) -> None:
    """
    Animasi scatter plot: titik data muncul satu per satu.
    Garis regresi digambar setelah semua titik selesai.

    Args:
        df    : DataFrame bersih.
        speed : Jeda antar titik (detik).
    """
    n = len(df)
    x_all = np.array(df["penjualan_bersih"].values, dtype=float) / 1e9
    y_all = np.array(df[TARGET].values, dtype=float) / 1e9
    q_all = (
        np.array(df["kuartal"].values, dtype=float)
        if "kuartal" in df.columns
        else np.ones(n)
    )
    labels_p = (
        df["periode"].tolist()
        if "periode" in df.columns
        else [str(i) for i in range(n)]
    )

    x_min, x_max = x_all.min() * 0.95, x_all.max() * 1.05
    y_min, y_max = y_all.min() * 0.90, y_all.max() * 1.10

    # Hitung garis regresi dari seluruh data (ditampilkan di akhir)
    m_reg, b_reg = np.polyfit(x_all, y_all, 1)

    kanvas = st.empty()
    progress = st.progress(0, text="Memplot titik data...")

    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.set_theme(style="whitegrid", palette="muted")

        # Titik yang sudah digambar (warna penuh)
        sc = ax.scatter(
            x_all[:i],
            y_all[:i],
            c=q_all[:i],
            cmap="viridis",
            s=80,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.6,
            vmin=1,
            vmax=4,
        )
        plt.colorbar(sc, ax=ax, label="Kuartal (1=Q1 … 4=Q4)")

        # Label pada titik terbaru
        ax.annotate(
            f" {labels_p[i - 1]}",
            xy=(x_all[i - 1], y_all[i - 1]),
            fontsize=7.5,
            color="#2C3E50",
            fontweight="bold",
        )

        # Garis regresi muncul hanya setelah semua titik selesai
        if i == n:
            x_sorted = sorted(x_all)
            ax.plot(
                x_sorted,
                [m_reg * xv + b_reg for xv in x_sorted],
                "r--",
                alpha=0.65,
                lw=2,
                label=f"Tren: y = {m_reg:.3f}x + {b_reg:.3f}",
            )
            ax.legend(fontsize=9)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Penjualan Bersih (Miliar Rp)", fontsize=10)
        ax.set_ylabel("Total Beban Pajak (Miliar Rp)", fontsize=10)
        ax.set_title(
            f"Penjualan vs Total Beban Pajak — {labels_p[i - 1]}  [{i}/{n}]",
            fontweight="bold",
            fontsize=11,
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.0f}M"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.2f}M"))
        ax.grid(alpha=0.35)

        plt.tight_layout()
        kanvas.pyplot(fig, use_container_width=True)
        plt.close(fig)
        progress.progress(i / n, text=f"Titik {labels_p[i - 1]}  ({i}/{n})")
        time.sleep(speed)

    progress.empty()
    st.toast("✅ Animasi scatter selesai!", icon="🔵")


def animasi_evaluasi(
    model: "LinearRegression", df_scaled: pd.DataFrame, n_test: int, speed: float = 0.09
) -> None:
    """
    Animasi timeline Aktual vs Prediksi: garis aktual digambar dulu,
    kemudian garis prediksi muncul bersamaan titik demi titik.

    Args:
        model    : LinearRegression yang sudah dilatih.
        df_scaled: DataFrame fitur + target (scaled).
        n_test   : Ukuran test set.
        speed    : Jeda antar frame (detik).
    """
    y_actual = np.array(df_scaled[TARGET].values, dtype=float)
    y_pred = np.array(
        model.predict(np.array(df_scaled[FEATURES].values, dtype=float)), dtype=float
    )
    n = len(df_scaled)
    n_train = n - n_test
    x_all = list(range(n))
    labels = (
        df_scaled["periode"].tolist()
        if "periode" in df_scaled.columns
        else [str(i) for i in x_all]
    )
    step_x = max(1, n // 9)

    y_min = min(y_actual.min(), y_pred.min()) / 1e9 * 0.93
    y_max = max(y_actual.max(), y_pred.max()) / 1e9 * 1.10

    kanvas = st.empty()
    progress = st.progress(0, text="Menggambar garis aktual...")

    # ── Fase 1: Gambar garis AKTUAL dulu (penuh, cepat) ─────────────────
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        sns.set_theme(style="whitegrid", palette="muted")

        ax.plot(
            x_all[:i],
            y_actual[:i] / 1e9,
            "o-",
            color="#2C3E50",
            lw=2.2,
            ms=5,
            label="Aktual",
        )

        # Garis pemisah train/test
        ax.axvline(x=n_train - 0.5, color="#95a5a6", linestyle=":", lw=2)
        ax.text(
            n_train - 0.3,
            y_min * 1.02,
            "← Train | Test →",
            fontsize=8,
            color="#7f8c8d",
            va="bottom",
        )

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_all[::step_x])
        ax.set_xticklabels(labels[::step_x], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Beban Pajak (Miliar Rp)", fontsize=10)
        ax.set_title(
            f"Aktual vs Prediksi — Menggambar data aktual [{i}/{n}]", fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))
        ax.grid(axis="y", alpha=0.35)

        plt.tight_layout()
        kanvas.pyplot(fig, use_container_width=True)
        plt.close(fig)
        progress.progress(i / (2 * n), text=f"Aktual [{i}/{n}]")
        time.sleep(speed * 0.5)  # Fase 1 lebih cepat

    # ── Fase 2: Overlay garis PREDIKSI satu per satu ────────────────────
    for i in range(1, n + 1):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        sns.set_theme(style="whitegrid", palette="muted")

        # Garis aktual selalu penuh
        ax.plot(
            x_all, y_actual / 1e9, "o-", color="#2C3E50", lw=2.2, ms=5, label="Aktual"
        )
        # Garis prediksi tumbuh
        ax.plot(
            x_all[:i],
            y_pred[:i] / 1e9,
            "s--",
            color="#E74C3C",
            lw=2.2,
            ms=5,
            label="Prediksi Model",
        )

        # Sorot selisih (error) pada titik terkini
        err = abs(y_actual[i - 1] - y_pred[i - 1]) / 1e9
        ax.vlines(
            i - 1,
            min(y_actual[i - 1], y_pred[i - 1]) / 1e9,
            max(y_actual[i - 1], y_pred[i - 1]) / 1e9,
            color="#F39C12",
            lw=2,
            alpha=0.7,
        )
        ax.annotate(
            f"Δ Rp{err:.2f}M",
            xy=(i - 1, (y_actual[i - 1] + y_pred[i - 1]) / 2 / 1e9),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7.5,
            color="#F39C12",
            fontweight="bold",
        )

        ax.axvline(x=n_train - 0.5, color="#95a5a6", linestyle=":", lw=2)
        ax.text(
            n_train - 0.3,
            y_min * 1.02,
            "← Train | Test →",
            fontsize=8,
            color="#7f8c8d",
            va="bottom",
        )

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_all[::step_x])
        ax.set_xticklabels(labels[::step_x], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Beban Pajak (Miliar Rp)", fontsize=10)
        ax.set_title(
            f"Prediksi Model muncul titik demi titik — {labels[i - 1]}  [{i}/{n}]",
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rp{v:.1f}M"))
        ax.grid(axis="y", alpha=0.35)

        plt.tight_layout()
        kanvas.pyplot(fig, use_container_width=True)
        plt.close(fig)
        progress.progress((n + i) / (2 * n), text=f"Prediksi [{i}/{n}]")
        time.sleep(speed)

    progress.empty()
    st.toast("✅ Animasi evaluasi selesai!", icon="🤖")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FUNGSI UTAMA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Fungsi utama yang merender seluruh antarmuka dashboard Streamlit.

    Struktur UI:
      ┌── Header ────────────────────────────────────────────────────────────┐
      │   Judul + subtitle + divider                                         │
      ├── Sidebar ───────────────────────────────────────────────────────────┤
      │   Slider input estimasi keuangan + info regulasi + tombol reset      │
      ├── Tab 1: Overview & Prediksi ────────────────────────────────────────┤
      │   KPI historis → Tren chart → Prediksi interaktif                   │
      ├── Tab 2: Analisis EDA ────────────────────────────────────────────────┤
      │   Statistik deskriptif → Heatmap → Boxplot → Scatter                │
      ├── Tab 3: Evaluasi Model ──────────────────────────────────────────────┤
      │   Metrik MAE/RMSE/R²/MAPE → Plot aktual vs prediksi → Koefisien     │
      └── Tab 4: Data Historis ──────────────────────────────────────────────┘
          Filter interaktif → Tabel → Ringkasan tahunan → Download CSV
    """

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="text-align:center; color:#2C3E50; margin-bottom:4px;">
            🏦 Dashboard Prediksi Beban Pajak
        </h1>
        <p style="text-align:center; color:#7F8C8D; font-size:15px; margin-top:0;">
            Estimasi <b>PPh Badan (22%)</b> &amp; <b>PPN Keluaran (11%)</b>
            — Kuartal Mendatang<br>
            <small>📋 Regulasi: UU HPP No.7 Tahun 2021 &nbsp;|&nbsp;
            Model: Linear Regression &nbsp;|&nbsp;
            Data: Historis 2015–2023</small>
        </p>
        <hr style="margin:10px 0 20px 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── MUAT ARTEFAK PIPELINE ──────────────────────────────────────────────────
    # ⚠️  Semua perintah st.* HARUS berada di LUAR get_artefak() untuk
    #     menghindari CacheReplayClosureError pada @st.cache_resource.
    #     Cek keberadaan artefak di sini, lalu tampilkan notifikasi,
    #     BARU kemudian panggil fungsi yang di-cache.
    artefak_baru = not os.path.exists(MODEL_PATH)
    if artefak_baru:
        # ─── ROOT CAUSE FIX ──────────────────────────────────────────────────
        # @st.cache_resource menyimpan hasil di MEMORI server, bukan di disk.
        # Menghapus model_artefak.pkl dari disk TIDAK otomatis menghapus cache
        # memori. Akibatnya get_artefak() masih mengembalikan data lama.
        #
        # Solusi: paksa clear cache memori SETIAP KALI file pkl tidak ada,
        # sehingga get_artefak() pasti dijalankan ulang dari awal.
        # ─────────────────────────────────────────────────────────────────────
        st.cache_resource.clear()
        st.info(
            "🔄 **Artefak belum ada.** Pipeline akan melatih model baru secara otomatis..."
        )
    with st.spinner("⏳ Memuat model dan data historis..."):
        artefak = get_artefak()
    if artefak_baru:
        st.toast("✅ Model berhasil dilatih dan disimpan!", icon="🤖")

    # Ekstrak komponen dari artefak
    model = artefak["model"]
    scaler = artefak["scaler"]
    df_raw_clean = artefak["df_raw_clean"]  # Data bersih sebelum scaling
    df_scaled = artefak["df_scaled"]  # Data bersih setelah scaling
    metrics = artefak["metrics"]
    n_test = max(1, int(len(df_scaled) * 0.20))  # Perkiraan ukuran test set

    # ── SIDEBAR — PANEL KONTROL INPUT ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Panel Prediksi Pajak")
        st.markdown(
            "Masukkan **estimasi keuangan kuartal mendatang** "
            "menggunakan slider di bawah ini."
        )
        st.divider()

        # Ambil nilai kuartal terakhir sebagai nilai awal slider
        last_penj = float(df_raw_clean["penjualan_bersih"].iloc[-1])
        last_hpp = float(df_raw_clean["hpp"].iloc[-1])
        last_ops = float(df_raw_clean["beban_operasional"].iloc[-1])

        # ── Slider: Penjualan Bersih ─────────────────────────────────────────
        penj_miliar = st.slider(
            "💰 Penjualan Bersih (Miliar Rp)",
            min_value=1.0,
            max_value=25.0,
            value=round(last_penj / 1e9, 1),
            step=0.1,
            help="Pendapatan bersih setelah retur & diskon pelanggan",
        )
        penjualan_input = penj_miliar * 1e9  # Konversi kembali ke Rupiah penuh

        # ── Slider: HPP / COGS ────────────────────────────────────────────────
        hpp_miliar = st.slider(
            "🏭 HPP / COGS (Miliar Rp)",
            min_value=0.5,
            max_value=18.0,
            value=round(last_hpp / 1e9, 1),
            step=0.1,
            help="Harga Pokok Penjualan: biaya langsung produksi atau pembelian barang",
        )
        hpp_input = hpp_miliar * 1e9

        # ── Slider: Beban Operasional ─────────────────────────────────────────
        ops_miliar = st.slider(
            "📋 Beban Operasional (Miliar Rp)",
            min_value=0.1,
            max_value=6.0,
            value=round(last_ops / 1e9, 2),
            step=0.05,
            help="Gaji karyawan, sewa kantor, utilitas, pemasaran, dan administrasi",
        )
        ops_input = ops_miliar * 1e9

        st.divider()

        # ── Validasi laba ─────────────────────────────────────────────────────
        laba_proyeksi = penjualan_input - hpp_input - ops_input
        if laba_proyeksi <= 0:
            st.warning(
                f"⚠️ **Proyeksi RUGI**: HPP + Ops ≥ Penjualan\n\n"
                f"Laba = **{fmt_rp(laba_proyeksi, singkat=True)}**\n\n"
                f"PPh Badan = **Rp 0** (tidak ada pajak penghasilan saat rugi)"
            )
        else:
            st.success(
                f"✅ **Proyeksi LABA**: {fmt_rp(laba_proyeksi, singkat=True)}\n\n"
                f"PPh = 22% × {fmt_rp(laba_proyeksi, singkat=True)} "
                f"= **{fmt_rp(laba_proyeksi * TAX_PPH_RATE, singkat=True)}**"
            )

        st.divider()

        # ── Info Regulasi ─────────────────────────────────────────────────────
        st.markdown("### ℹ️ Regulasi Perpajakan")
        st.info(
            f"**PPh Badan** : {TAX_PPH_RATE * 100:.0f}% × Laba Kena Pajak\n\n"
            f"**PPN Keluaran** : {TAX_PPN_RATE * 100:.0f}% × Penjualan Bersih\n\n"
            f"📌 *Sumber: UU HPP No.7 Tahun 2021*"
        )

        st.divider()

        # ── Tombol reset model ────────────────────────────────────────────────
        st.markdown("### 🔄 Manajemen Model")
        if st.button(
            "🗑️ Hapus Cache & Latih Ulang",
            use_container_width=True,
            help="Hapus artefak yang ada dan latih model baru dari awal",
        ):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            st.cache_resource.clear()
            st.rerun()  # Reload halaman → pipeline dilatih ulang

    # ── PANEL NAVIGASI TAB ────────────────────────────────────────────────────
    # ── PANEL NAVIGASI TAB ─────────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview & Prediksi",
            "📈 Analisis EDA",
            "🤖 Evaluasi Model",
            "📋 Data Historis",
            "📥 Input Data",
        ]
    )

    # =========================================================================
    # TAB 1 — OVERVIEW & PREDIKSI INTERAKTIF
    # =========================================================================
    with tab1:
        # ── KPI Historis ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">📊 Ringkasan KPI Historis Perusahaan</div>',
            unsafe_allow_html=True,
        )

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)

        total_pajak_hist = df_raw_clean[TARGET].sum()
        avg_pajak = df_raw_clean[TARGET].mean()
        total_ppn_hist = df_raw_clean["ppn_keluaran"].sum()
        total_pph_hist = df_raw_clean["pph_badan"].sum()

        # Hitung tren: bandingkan rata-rata 4 kuartal terakhir vs 4 kuartal sebelumnya
        n_rows = len(df_raw_clean)
        last4 = df_raw_clean[TARGET].iloc[-4:].mean() if n_rows >= 4 else avg_pajak
        prev4 = df_raw_clean[TARGET].iloc[-8:-4].mean() if n_rows >= 8 else avg_pajak
        delta_pct = (last4 - prev4) / prev4 * 100 if prev4 > 0 else 0.0

        with col_k1:
            st.metric(
                label="💰 Total Pajak Historis",
                value=fmt_rp(total_pajak_hist, singkat=True),
                delta=f"Rata-rata: {fmt_rp(avg_pajak, singkat=True)}/Kuartal",
                delta_color="off",
                help="Total akumulasi beban pajak seluruh periode historis",
            )
        with col_k2:
            st.metric(
                label="🔵 Total PPN (11%)",
                value=fmt_rp(total_ppn_hist, singkat=True),
                delta=f"{total_ppn_hist / total_pajak_hist * 100:.1f}% dari total",
                delta_color="off",
                help="Akumulasi PPN Keluaran = 11% × Penjualan Bersih",
            )
        with col_k3:
            st.metric(
                label="🟢 Total PPh (22%)",
                value=fmt_rp(total_pph_hist, singkat=True),
                delta=f"{total_pph_hist / total_pajak_hist * 100:.1f}% dari total",
                delta_color="off",
                help="Akumulasi PPh Badan = 22% × Laba Kena Pajak",
            )
        with col_k4:
            st.metric(
                label="📈 Tren Pajak (4 Kuartal Terakhir)",
                value=fmt_rp(last4, singkat=True),
                delta=f"{delta_pct:+.1f}% vs 4 kuartal lalu",
                delta_color="normal",
                help="Perbandingan rata-rata beban pajak 4 kuartal terbaru vs sebelumnya",
            )

        st.divider()

        # ── Tren Historis Chart ────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">📈 Tren Historis Beban Pajak per Kuartal</div>',
            unsafe_allow_html=True,
        )

        # ── Kontrol animasi ──────────────────────────────────────────────────
        c_btn1, c_spd1, c_info1 = st.columns([1, 2, 3])
        with c_btn1:
            btn_tren = st.button(
                "▶️ Putar Animasi",
                key="btn_tren",
                type="primary",
                use_container_width=True,
            )
        with c_spd1:
            spd_tren = st.slider(
                "Kecepatan (detik/frame)",
                0.02,
                0.40,
                0.07,
                0.01,
                key="spd_tren",
                format="%.2f",
            )
        with c_info1:
            st.caption(
                "💡 Klik **▶️ Putar Animasi** untuk melihat grafik digambar "
                "kuartal demi kuartal. Geser slider untuk mengatur kecepatan."
            )

        if btn_tren:
            # Mode animasi: grafik digambar progresif via st.empty()
            animasi_tren_pajak(df_raw_clean, speed=spd_tren)
        else:
            # Mode statis: tampilkan grafik lengkap sekaligus
            fig_tren = plot_tren_pajak(df_raw_clean)
            st.pyplot(fig_tren, use_container_width=True)
            plt.close(fig_tren)

        st.divider()

        # ── PREDIKSI INTERAKTIF ────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🔮 Prediksi Beban Pajak — Kuartal Mendatang</div>',
            unsafe_allow_html=True,
        )

        # Jalankan prediksi berdasarkan nilai slider saat ini
        hasil = predict_next_quarter(
            model=model,
            scaler=scaler,
            penjualan_bersih=penjualan_input,
            hpp=hpp_input,
            beban_operasional=ops_input,
        )

        col_p1, col_p2, col_p3 = st.columns([1.1, 1.1, 0.9])

        # Panel kiri: Ringkasan input
        with col_p1:
            st.markdown("#### 🧮 Input Estimasi Keuangan")
            tabel_input = pd.DataFrame(
                {
                    "Parameter": [
                        "Penjualan Bersih",
                        "HPP / COGS",
                        "Beban Operasional",
                        "─────────────────",
                        "Laba Kotor",
                        "Laba Sebelum Pajak",
                    ],
                    "Nilai": [
                        fmt_rp(penjualan_input, True),
                        fmt_rp(hpp_input, True),
                        fmt_rp(ops_input, True),
                        "─────────────",
                        fmt_rp(penjualan_input - hpp_input, True),
                        fmt_rp(hasil["laba_kena_pajak"], True),
                    ],
                }
            )
            st.dataframe(tabel_input, use_container_width=True, hide_index=True)

        # Panel tengah: Hasil prediksi
        with col_p2:
            st.markdown("#### 📊 Hasil Prediksi Model")
            tabel_pred = pd.DataFrame(
                {
                    "Komponen Pajak": [
                        "🔵 PPN Keluaran (11%)",
                        "🟢 PPh Badan (22%)",
                        "─────────────────────",
                        "🔴 Total Beban Pajak",
                        "📉 Effective Tax Rate",
                    ],
                    "Estimasi": [
                        fmt_rp(hasil["ppn_keluaran"], True),
                        fmt_rp(hasil["pph_badan"], True),
                        "─────────────────────",
                        fmt_rp(hasil["total_beban_pajak"], True),
                        f"{hasil['effective_tax_rate']:.2f}%",
                    ],
                }
            )
            st.dataframe(tabel_pred, use_container_width=True, hide_index=True)

            # Status laba/rugi
            if hasil["laba_kena_pajak"] <= 0:
                st.error(
                    "⚠️ **Kondisi RUGI** — PPh Badan = Rp 0\n\n"
                    "Sesuai regulasi, PPh hanya dikenakan jika laba positif."
                )
            else:
                st.success(
                    f"✅ **Laba Kena Pajak**: {fmt_rp(hasil['laba_kena_pajak'], True)}"
                )

        # Panel kanan: Pie chart komposisi
        with col_p3:
            st.markdown("#### 🥧 Komposisi Pajak")
            fig_pie = plot_pie_komposisi(
                hasil["ppn_keluaran"], hasil["pph_badan"], "Kuartal Mendatang"
            )
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)

        # ── Banner Total Prediksi ─────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="prediction-banner">
                <h3>🧾 Estimasi Total Beban Pajak Kuartal Mendatang</h3>
                <h1>{fmt_rp(hasil["total_beban_pajak"])}</h1>
                <p>
                    PPN Keluaran: {fmt_rp(hasil["ppn_keluaran"], True)}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    PPh Badan: {fmt_rp(hasil["pph_badan"], True)}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Effective Rate: {hasil["effective_tax_rate"]:.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Catatan disclaimer
        st.markdown(
            """
            <div class="warn-box">
            ⚠️ <b>Disclaimer:</b> Angka di atas adalah <i>estimasi</i> berbasis model statistik
            dan hanya untuk keperluan perencanaan internal. Selalu konsultasikan dengan
            <b>konsultan pajak bersertifikat (Brevet A/B)</b> untuk kewajiban pajak resmi.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2 — ANALISIS EDA
    # =========================================================================
    with tab2:
        st.markdown(
            '<div class="section-header">🔍 Exploratory Data Analysis (EDA)</div>',
            unsafe_allow_html=True,
        )

        # ── Statistik Deskriptif ──────────────────────────────────────────────
        st.markdown("#### 📋 Statistik Deskriptif (satuan: Miliar Rupiah)")

        desc_cols = FEATURES + [TARGET, "ppn_keluaran", "pph_badan"]
        desc = df_raw_clean[desc_cols].describe().T

        # Rename baris untuk keterbacaan
        desc.index = [
            "Penjualan Bersih",
            "HPP / COGS",
            "Beban Operasional",
            "Total Pajak",
            "PPN Keluaran",
            "PPh Badan",
        ]
        # Konversi ke Miliar Rupiah agar mudah dibaca
        num_cols = ["mean", "std", "min", "25%", "50%", "75%", "max"]
        desc_show = desc.copy()
        for col in num_cols:
            if col in desc_show.columns:
                desc_show[col] = (desc_show[col] / 1e9).round(3)
        desc_show["count"] = desc_show["count"].astype(int)

        st.dataframe(
            desc_show.style.format({c: "{:.3f}" for c in num_cols}).background_gradient(
                cmap="YlOrRd", subset=["mean"]
            ),
            use_container_width=True,
        )
        st.caption("Satuan kolom numerik: Miliar Rupiah (Rp)")

        st.divider()

        # ── Heatmap + Boxplot (dua kolom) ─────────────────────────────────────
        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown("#### 🌡️ Heatmap Korelasi")
            fig_heat = plot_heatmap_korelasi(df_raw_clean)
            st.pyplot(fig_heat, use_container_width=True)
            plt.close(fig_heat)
            st.markdown(
                """
                <div class="info-box">
                📌 <b>Cara baca:</b> Nilai mendekati <b>+1.0</b> = korelasi positif kuat
                (jika A naik, B juga naik). Penjualan → PPN memiliki korelasi sempurna
                karena PPN = 11% × Penjualan secara langsung (deterministic).
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_e2:
            st.markdown("#### 📦 Distribusi Pajak per Kuartal")
            if "kuartal" in df_raw_clean.columns:
                fig_box = plot_boxplot_kuartal(df_raw_clean)
                st.pyplot(fig_box, use_container_width=True)
                plt.close(fig_box)
                st.markdown(
                    """
                    <div class="info-box">
                    📌 <b>Insight musiman:</b> <b>Q4 (Okt–Des)</b> secara konsisten
                    memiliki beban pajak tertinggi karena peningkatan penjualan akhir tahun
                    (Harbolnas, Natal, Tahun Baru). <b>Q1</b> biasanya paling rendah
                    karena awal tahun cenderung lebih sepi.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Scatter Penjualan vs Pajak ──────────────────────────────────────────────────
        st.markdown("#### 🔵 Scatter: Penjualan Bersih vs Total Beban Pajak")

        c_btn2, c_spd2, c_info2 = st.columns([1, 2, 3])
        with c_btn2:
            btn_sc = st.button(
                "▶️ Putar Animasi",
                key="btn_sc",
                type="primary",
                use_container_width=True,
            )
        with c_spd2:
            spd_sc = st.slider(
                "Kecepatan (detik/titik)",
                0.02,
                0.50,
                0.10,
                0.01,
                key="spd_sc",
                format="%.2f",
            )
        with c_info2:
            st.caption(
                "💡 Titik data muncul satu per satu per kuartal. "
                "Garis regresi muncul otomatis setelah semua titik selesai."
            )

        if btn_sc:
            animasi_scatter(df_raw_clean, speed=spd_sc)
        else:
            fig_sc = plot_scatter_penjualan_pajak(df_raw_clean)
            st.pyplot(fig_sc, use_container_width=True)
            plt.close(fig_sc)
        st.markdown(
            """
            <div class="info-box">
            📌 <b>Interpretasi:</b> Titik-titik yang mengikuti garis tren merah (putus-putus)
            menunjukkan hubungan linear yang kuat. Warna titik menunjukkan kuartal —
            titik gelap (Q4) cenderung berada di kanan atas (penjualan & pajak tinggi).
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 3 — EVALUASI MODEL
    # =========================================================================
    with tab3:
        st.markdown(
            '<div class="section-header">🤖 Performa Model Linear Regression</div>',
            unsafe_allow_html=True,
        )

        # ── Kartu Metrik ──────────────────────────────────────────────────────
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric(
                "📉 MAE",
                fmt_rp(metrics["MAE"], singkat=True),
                help="Mean Absolute Error: Rata-rata kesalahan absolut prediksi vs aktual",
            )
        with col_m2:
            st.metric(
                "📉 RMSE",
                fmt_rp(metrics["RMSE"], singkat=True),
                help="Root Mean Squared Error: Lebih sensitif terhadap prediksi yang sangat meleset",
            )
        with col_m3:
            r2 = metrics["R2"]
            st.metric(
                "📈 R² Score",
                f"{r2:.4f}",
                delta=f"{r2 * 100:.1f}% variansi dijelaskan",
                delta_color="normal" if r2 > 0 else "inverse",
                help="R-Squared: 1.0 = sempurna | 0.0 = sama rata-rata | <0 = buruk",
            )
        with col_m4:
            st.metric(
                "📊 MAPE",
                f"{metrics['MAPE']:.2f}%",
                help="Mean Absolute Percentage Error: Error dalam persentase, mudah dibandingkan",
            )

        st.divider()

        # ── Interpretasi Otomatis R² ───────────────────────────────────────────
        r2 = metrics["R2"]
        if r2 >= 0.90:
            st.success(
                f"🟢 **SANGAT BAIK** — R² = {r2:.4f}\n\n"
                f"Model mampu menjelaskan **{r2 * 100:.1f}%** variansi data beban pajak. "
                f"Siap digunakan untuk estimasi internal."
            )
        elif r2 >= 0.75:
            st.info(
                f"🟡 **BAIK** — R² = {r2:.4f}\n\n"
                f"Model menjelaskan **{r2 * 100:.1f}%** variansi. "
                f"Disarankan validasi tambahan sebelum deployment produksi."
            )
        elif r2 >= 0.50:
            st.warning(
                f"🟠 **CUKUP** — R² = {r2:.4f}\n\n"
                f"Model menjelaskan **{r2 * 100:.1f}%** variansi. "
                f"Pertimbangkan feature engineering atau tambah fitur baru."
            )
        else:
            st.error(
                f"🔴 **KURANG** — R² = {r2:.4f}\n\n"
                f"Coba model yang lebih kompleks: Random Forest, Gradient Boosting, "
                f"atau Polynomial Regression."
            )

        st.divider()

        # ── Plot Aktual vs Prediksi ────────────────────────────────────────────────────
        st.markdown("#### 📈 Perbandingan Aktual vs Prediksi Model")

        c_btn3, c_spd3, c_info3 = st.columns([1, 2, 3])
        with c_btn3:
            btn_eval = st.button(
                "▶️ Putar Animasi",
                key="btn_eval",
                type="primary",
                use_container_width=True,
            )
        with c_spd3:
            spd_eval = st.slider(
                "Kecepatan (detik/frame)",
                0.02,
                0.40,
                0.09,
                0.01,
                key="spd_eval",
                format="%.2f",
            )
        with c_info3:
            st.caption(
                "💡 **Fase 1**: garis aktual muncul dulu. "
                "**Fase 2**: garis prediksi overlay + error Δ tiap titik ditampilkan."
            )

        if btn_eval:
            animasi_evaluasi(model, df_scaled, n_test, speed=spd_eval)
        else:
            fig_eval = plot_aktual_vs_prediksi(model, df_scaled, n_test)
            st.pyplot(fig_eval, use_container_width=True)
            plt.close(fig_eval)

        st.divider()

        # ── Koefisien Model ────────────────────────────────────────────────────
        st.markdown("#### 🔢 Koefisien Model — Interpretasi Bisnis")

        coef_data = {
            "Komponen": ["Intercept (bias)"] + FEATURES,
            "Koefisien (scaled)": [model.intercept_] + list(model.coef_),
            "Arah Pengaruh": ["—"]
            + ["↑ Naik" if c > 0 else "↓ Turun" for c in model.coef_],
            "Interpretasi": [
                "Beban pajak dasar ketika semua fitur = 0 (dalam skala)",
                "Kontribusi Penjualan Bersih terhadap prediksi pajak",
                "Kontribusi HPP / COGS terhadap prediksi pajak",
                "Kontribusi Beban Operasional terhadap prediksi pajak",
            ],
        }
        df_coef = pd.DataFrame(coef_data)
        st.dataframe(
            df_coef.style.format({"Koefisien (scaled)": "{:,.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class="info-box">
            📌 <b>Catatan Interpretasi:</b> Koefisien di atas dalam unit <i>scaled</i>
            (StandardScaler: z = (x−μ)/σ). Magnitude koefisien mencerminkan kepentingan
            relatif fitur. Tanda positif (+) berarti kenaikan 1σ fitur tersebut
            → prediksi pajak naik sebesar nilai koefisien (dalam Rupiah).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Rekomendasi Peningkatan Model ─────────────────────────────────────
        with st.expander("💡 Rekomendasi Peningkatan Model (Klik untuk expand)"):
            st.markdown(
                """
                **Jika R² masih belum memuaskan, pertimbangkan langkah berikut:**

                | Langkah | Deskripsi | Potensi Dampak |
                |---------|-----------|----------------|
                | Feature Engineering | Tambahkan fitur: pertumbuhan YoY, rasio HPP/Penjualan, inflasi | ↑ Tinggi |
                | Polynomial Features | Tambahkan term kuadratik untuk menangkap hubungan non-linear | ↑ Sedang |
                | Random Forest | Ensemble method yang mampu menangkap interaksi antar fitur | ↑ Tinggi |
                | Gradient Boosting | XGBoost/LightGBM — sangat efektif untuk data keuangan | ↑ Sangat Tinggi |
                | Data Lebih Banyak | Gunakan data bulanan (bukan kuartalan) untuk lebih banyak sampel | ↑ Tinggi |
                | Time Series Model | SARIMA / Prophet untuk menangkap komponen musiman secara eksplisit | ↑ Sedang |

                > 💡 Selalu lakukan **cross-validation time-series** (TimeSeriesSplit)
                > untuk evaluasi yang lebih robust pada data finansial.
                """
            )

    # =========================================================================
    # TAB 4 — DATA HISTORIS
    # =========================================================================
    with tab4:
        st.markdown(
            '<div class="section-header">📋 Data Historis Keuangan (Sudah Dibersihkan)</div>',
            unsafe_allow_html=True,
        )

        # ── Filter Interaktif ─────────────────────────────────────────────────
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if "tahun" in df_raw_clean.columns:
                all_years = sorted(df_raw_clean["tahun"].unique().tolist())
                sel_years = st.multiselect(
                    "🗓️ Filter Tahun:", all_years, default=all_years
                )
            else:
                sel_years = None
        with col_f2:
            if "kuartal" in df_raw_clean.columns:
                all_qtrs = sorted(df_raw_clean["kuartal"].unique().tolist())
                sel_qtrs = st.multiselect(
                    "📅 Filter Kuartal:", all_qtrs, default=all_qtrs
                )
            else:
                sel_qtrs = None

        # Terapkan filter
        df_disp = df_raw_clean.copy()
        if sel_years and "tahun" in df_disp.columns:
            df_disp = df_disp[df_disp["tahun"].isin(sel_years)]
        if sel_qtrs and "kuartal" in df_disp.columns:
            df_disp = df_disp[df_disp["kuartal"].isin(sel_qtrs)]

        # ── Siapkan tampilan tabel ────────────────────────────────────────────
        disp_cols = [
            "periode",
            "tahun",
            "kuartal",
            "penjualan_bersih",
            "hpp",
            "beban_operasional",
            "ppn_keluaran",
            "pph_badan",
            "total_beban_pajak",
        ]
        avail = [c for c in disp_cols if c in df_disp.columns]
        df_show = df_disp[avail].copy()

        # Rename kolom untuk keterbacaan
        rename = {
            "periode": "Periode",
            "tahun": "Tahun",
            "kuartal": "Q",
            "penjualan_bersih": "Penjualan (Jt)",
            "hpp": "HPP (Jt)",
            "beban_operasional": "Bbn.Ops (Jt)",
            "ppn_keluaran": "PPN (Jt)",
            "pph_badan": "PPh (Jt)",
            "total_beban_pajak": "Total Pajak (Jt)",
        }
        df_show.rename(
            columns={k: v for k, v in rename.items() if k in df_show.columns},
            inplace=True,
        )

        # Konversi ke Juta Rupiah agar angka lebih ringkas di tabel
        juta_cols = [c for c in df_show.columns if "(Jt)" in c]
        for col in juta_cols:
            df_show[col] = (df_show[col] / 1e6).round(1)

        st.dataframe(
            df_show.style.format({c: "{:,.1f}" for c in juta_cols}).background_gradient(
                cmap="YlOrRd",
                subset=["Total Pajak (Jt)"]
                if "Total Pajak (Jt)" in df_show.columns
                else [],
            ),
            use_container_width=True,
            height=420,
        )
        st.caption(
            f"📌 Menampilkan **{len(df_show)}** dari **{len(df_raw_clean)}** baris | "
            f"Satuan kolom angka: **Juta Rupiah (Jt)**"
        )

        # ── Tombol Download CSV ───────────────────────────────────────────────
        csv_bytes = df_raw_clean.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Data Historis (CSV)",
            data=csv_bytes,
            file_name="data_historis_pajak_bersih.csv",
            mime="text/csv",
            use_container_width=True,
            help="Unduh data historis yang sudah dibersihkan dalam format CSV",
        )

        st.divider()

        # ── Ringkasan per Tahun ───────────────────────────────────────────────
        st.markdown("#### 📊 Ringkasan Agregat Tahunan")
        if "tahun" in df_raw_clean.columns:
            summary = (
                df_raw_clean.groupby("tahun")
                .agg(
                    Penjualan_M=("penjualan_bersih", "sum"),
                    HPP_M=("hpp", "sum"),
                    Ops_M=("beban_operasional", "sum"),
                    PPN_M=("ppn_keluaran", "sum"),
                    PPh_M=("pph_badan", "sum"),
                    Total_Pajak_M=(TARGET, "sum"),
                )
                .reset_index()
            )
            # Konversi ke Miliar Rupiah
            for col in summary.columns[1:]:
                summary[col] = (summary[col] / 1e9).round(2)
            summary.rename(
                columns={"tahun": "Tahun"},
                inplace=True,
            )

            st.dataframe(
                summary.style.format(
                    {c: "{:.2f}" for c in summary.columns[1:]}
                ).background_gradient(
                    cmap="YlOrRd",
                    subset=["Total_Pajak_M"]
                    if "Total_Pajak_M" in summary.columns
                    else [],
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Satuan: Miliar Rupiah (M)")

        st.markdown(
            """
            <div class="info-box">
            📌 <b>Catatan:</b> Data ini merupakan hasil setelah proses
            <b>Preprocessing</b>: duplikat dihapus, missing values diimputasi
            dengan median, dan outlier (IQR) dibuang. Data yang ditampilkan
            adalah representasi bersih yang digunakan sebagai input model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 5 — INPUT DATA
    # =========================================================================
    with tab5:
        st.markdown(
            '<div class="section-header">📥 Manajemen Data Keuangan</div>',
            unsafe_allow_html=True,
        )

        # ── STATUS DATA SAAT INI ────────────────────────────────────────────────
        ada_data_user = os.path.exists(USER_DATA_PATH)
        if ada_data_user:
            st.success(
                f"✅ **Menggunakan data Anda** dari `{USER_DATA_PATH}` — "
                f"{len(df_raw_clean)} baris setelah preprocessing."
            )
        else:
            st.info(
                "🔵 **Menggunakan data simulasi ERP.** "
                "Upload CSV atau isi form di bawah untuk mengganti dengan data nyata."
            )

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # BAGIAN A — UPLOAD CSV
        # ──────────────────────────────────────────────────────────────────────
        st.markdown("### 📂 A. Upload File CSV")

        col_dl, col_fmt = st.columns([1, 3])
        with col_dl:
            # Tombol unduh template CSV
            st.download_button(
                label="⬇️ Download Template CSV",
                data=buat_template_csv(),
                file_name="template_data_pajak.csv",
                mime="text/csv",
                use_container_width=True,
                help="Unduh file contoh, isi dengan data nyata, lalu upload kembali.",
            )
        with col_fmt:
            st.markdown(
                """
                <div class="info-box">
                📌 <b>Format kolom wajib di CSV:</b><br>
                <code>tahun</code> (int) &nbsp;|
                <code>kuartal</code> (1–4) &nbsp;|
                <code>penjualan_bersih</code> (Rp) &nbsp;|
                <code>hpp</code> (Rp) &nbsp;|
                <code>beban_operasional</code> (Rp)<br>
                <small>Kolom lain akan dihitung otomatis (PPN, PPh, total pajak).</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

        uploaded = st.file_uploader(
            "Pilih file CSV data keuangan:",
            type=["csv"],
            help="Format: tahun, kuartal, penjualan_bersih, hpp, beban_operasional",
        )

        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded, encoding="utf-8")
            except Exception:
                df_upload = pd.read_csv(uploaded, encoding="latin-1")

            # Validasi
            errors_upload = validasi_input_data(df_upload)

            if errors_upload:
                for err in errors_upload:
                    st.error(f"❌ {err}")
            else:
                df_preview = hitung_kolom_pajak(df_upload)
                st.success(
                    f"✅ File valid! {len(df_preview)} baris ditemukan. "
                    "Preview di bawah — klik **Simpan & Latih Ulang** untuk menerapkan."
                )

                # Preview tabel
                st.dataframe(
                    df_preview[
                        [
                            "periode",
                            "tahun",
                            "kuartal",
                            "penjualan_bersih",
                            "hpp",
                            "beban_operasional",
                            "ppn_keluaran",
                            "pph_badan",
                            "total_beban_pajak",
                        ]
                    ]
                    .style.format(
                        {
                            c: "{:,.0f}"
                            for c in [
                                "penjualan_bersih",
                                "hpp",
                                "beban_operasional",
                                "ppn_keluaran",
                                "pph_badan",
                                "total_beban_pajak",
                            ]
                        }
                    )
                    .highlight_max(subset=["total_beban_pajak"], color="#fff3cd"),  # type: ignore[arg-type]
                    use_container_width=True,
                    height=280,
                )

                if st.button(
                    "💾 Simpan & Latih Ulang Model",
                    type="primary",
                    use_container_width=True,
                    key="btn_simpan_upload",
                ):
                    save_user_data(df_preview)
                    if os.path.exists(MODEL_PATH):
                        os.remove(MODEL_PATH)
                    st.cache_resource.clear()
                    st.success("✅ Data disimpan! Melatih ulang model...")
                    st.rerun()

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # BAGIAN B — INPUT MANUAL (TAMBAH KUARTAL BARU)
        # ──────────────────────────────────────────────────────────────────────
        st.markdown("### ✏️ B. Input Manual Kuartal Baru")
        st.caption(
            "Tambahkan satu kuartal baru ke dataset yang sudah ada. "
            "Jika belum ada data, kuartal ini menjadi baris pertama."
        )

        with st.form("form_tambah_kuartal", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns(3)

            with fc1:
                f_tahun = st.number_input(
                    "Tahun",
                    min_value=2000,
                    max_value=2035,
                    value=2024,
                    step=1,
                )
                f_kuartal = st.selectbox(
                    "Kuartal",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: (
                        f"Q{x} — " + ["Jan–Mar", "Apr–Jun", "Jul–Sep", "Okt–Des"][x - 1]
                    ),
                )

            with fc2:
                f_penjualan = (
                    st.number_input(
                        "💰 Penjualan Bersih (Juta Rp)",
                        min_value=0.0,
                        value=8_500.0,
                        step=100.0,
                        format="%.1f",
                        help="Masukkan dalam satuan Juta Rupiah",
                    )
                    * 1_000_000
                )  # konversi ke Rupiah penuh

                f_hpp = (
                    st.number_input(
                        "🏭 HPP / COGS (Juta Rp)",
                        min_value=0.0,
                        value=5_100.0,
                        step=100.0,
                        format="%.1f",
                    )
                    * 1_000_000
                )

            with fc3:
                f_ops = (
                    st.number_input(
                        "📋 Beban Operasional (Juta Rp)",
                        min_value=0.0,
                        value=1_250.0,
                        step=50.0,
                        format="%.1f",
                    )
                    * 1_000_000
                )

                # Kalkulasi preview pajak langsung
                laba_prev = max(0, f_penjualan - f_hpp - f_ops)
                ppn_prev = f_penjualan * TAX_PPN_RATE
                pph_prev = laba_prev * TAX_PPH_RATE
                total_prev = ppn_prev + pph_prev

                st.markdown(
                    f"""
                    <div class="success-box">
                    🧮 <b>Preview Pajak:</b><br>
                    PPN : <b>{fmt_rp(ppn_prev, True)}</b><br>
                    PPh : <b>{fmt_rp(pph_prev, True)}</b><br>
                    Total : <b>{fmt_rp(total_prev, True)}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            submitted_manual = st.form_submit_button(
                "➕ Tambah Kuartal Ini ke Dataset",
                type="primary",
                use_container_width=True,
            )

        if submitted_manual:
            baris_baru = pd.DataFrame(
                [
                    {
                        "tahun": int(f_tahun),
                        "kuartal": int(f_kuartal),
                        "penjualan_bersih": f_penjualan,
                        "hpp": f_hpp,
                        "beban_operasional": f_ops,
                    }
                ]
            )

            # Gabungkan dengan data yang sudah ada (jika ada)
            if os.path.exists(USER_DATA_PATH):
                df_lama = pd.read_csv(USER_DATA_PATH, encoding="utf-8")
                df_gabung = pd.concat(
                    [
                        df_lama[
                            [c for c in REQUIRED_INPUT_COLS if c in df_lama.columns]
                        ],
                        baris_baru,
                    ],
                    ignore_index=True,
                )
            else:
                df_gabung = baris_baru

            # Hapus duplikat periode (tahun+kuartal sama)
            df_gabung = pd.DataFrame(df_gabung).drop_duplicates().reset_index(drop=True)
            df_gabung = (
                df_gabung.sort_values("tahun")
                .sort_values("kuartal")
                .reset_index(drop=True)
            )

            errors_manual = validasi_input_data(df_gabung)
            if errors_manual:
                for err in errors_manual:
                    st.warning(f"⚠️ {err}")
                st.info(
                    "Data tetap ditambahkan. Latih ulang model setelah "
                    "jumlah data mencukupi (≥ 8 kuartal)."
                )

            df_final = hitung_kolom_pajak(df_gabung)
            save_user_data(df_final)

            st.success(
                f"✅ Kuartal **{int(f_tahun)}-Q{int(f_kuartal)}** berhasil ditambahkan! "
                f"Total dataset: **{len(df_gabung)} baris**."
            )

            if len(df_gabung) >= 8:
                if st.button(
                    "🔄 Latih Ulang Model Sekarang",
                    type="primary",
                    key="btn_latih_manual",
                ):
                    if os.path.exists(MODEL_PATH):
                        os.remove(MODEL_PATH)
                    st.cache_resource.clear()
                    st.rerun()
            else:
                sisa = 8 - len(df_gabung)
                st.info(
                    f"📌 Tambahkan {sisa} kuartal lagi sebelum model dapat dilatih."
                )

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # BAGIAN C — EDIT & HAPUS DATA YANG ADA
        # ──────────────────────────────────────────────────────────────────────
        st.markdown("### 📤 C. Edit atau Hapus Data yang Ada")

        if os.path.exists(USER_DATA_PATH):
            df_edit_raw = pd.read_csv(USER_DATA_PATH, encoding="utf-8")
            input_cols_show = [
                c for c in REQUIRED_INPUT_COLS if c in df_edit_raw.columns
            ]

            st.caption(
                "Edit langsung di tabel, lalu klik **Simpan Perubahan**. "
                "Untuk hapus baris: pilih baris (centang) lalu klik **Hapus Baris Dipilih**."
            )

            df_editable_raw = st.data_editor(
                df_edit_raw[input_cols_show],
                use_container_width=True,
                num_rows="dynamic",  # Izinkan tambah/hapus baris
                column_config={
                    "tahun": st.column_config.NumberColumn(
                        "Tahun", min_value=2000, max_value=2035, step=1, format="%d"
                    ),
                    "kuartal": st.column_config.SelectboxColumn(
                        "Kuartal", options=[1, 2, 3, 4]
                    ),
                    "penjualan_bersih": st.column_config.NumberColumn(
                        "Penjualan Bersih (Rp)", min_value=0, format="%.0f"
                    ),
                    "hpp": st.column_config.NumberColumn(
                        "HPP (Rp)", min_value=0, format="%.0f"
                    ),
                    "beban_operasional": st.column_config.NumberColumn(
                        "Bbn. Operasional (Rp)", min_value=0, format="%.0f"
                    ),
                },
                key="tabel_edit",
            )
            # Cast hasil st.data_editor ke DataFrame eksplisit
            df_editable = (
                pd.DataFrame(df_editable_raw)
                if df_editable_raw is not None
                else pd.DataFrame()
            )

            col_sv, col_dl2 = st.columns(2)
            with col_sv:
                if st.button(
                    "💾 Simpan Perubahan & Latih Ulang",
                    type="primary",
                    use_container_width=True,
                    key="btn_simpan_edit",
                ):
                    errors_edit = validasi_input_data(df_editable)
                    if errors_edit:
                        for err in errors_edit:
                            st.error(f"❌ {err}")
                    else:
                        df_saved = hitung_kolom_pajak(df_editable)
                        save_user_data(df_saved)
                        if os.path.exists(MODEL_PATH):
                            os.remove(MODEL_PATH)
                        st.cache_resource.clear()
                        st.success("✅ Perubahan disimpan! Melatih ulang model...")
                        st.rerun()
            with col_dl2:
                csv_user = df_edit_raw[input_cols_show].to_csv(
                    index=False, encoding="utf-8"
                )
                st.download_button(
                    "⬇️ Download Data Saat Ini (CSV)",
                    data=csv_user,
                    file_name="data_keuangan_saya.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        else:
            st.info(
                "🔵 Belum ada data pengguna. Upload CSV atau isi form manual di atas."
            )

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # BAGIAN D — RESET KE DATA SIMULASI
        # ──────────────────────────────────────────────────────────────────────
        st.markdown("### 🔄 D. Reset ke Data Simulasi ERP")
        st.markdown(
            """
            <div class="warn-box">
            ⚠️ Tombol ini akan <b>menghapus data Anda</b> dan mengembalikan dashboard
            ke dataset simulasi bawaan. Model akan dilatih ulang otomatis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_rst1, col_rst2 = st.columns([1, 3])
        with col_rst1:
            konfirmasi_reset = st.checkbox(
                "⚠️ Saya mengerti data saya akan dihapus",
                key="konfirmasi_reset",
            )
        with col_rst2:
            if st.button(
                "🗑️ Reset ke Data Simulasi",
                type="secondary",
                disabled=not konfirmasi_reset,
                use_container_width=False,
                key="btn_reset_data",
            ):
                if os.path.exists(USER_DATA_PATH):
                    os.remove(USER_DATA_PATH)
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                st.cache_resource.clear()
                st.success("✅ Data direset! Memuat ulang dengan data simulasi...")
                st.rerun()

    # ── FOOTER ─────────────────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#95a5a6; font-size:12px; padding:8px 0;">
            🏦 <b>Tax Prediction Dashboard</b> &nbsp;|&nbsp;
            Model: <i>Linear Regression</i> &nbsp;|&nbsp;
            Regulasi: UU HPP No.7 Tahun 2021 (PPh 22%, PPN 11%) &nbsp;|&nbsp;
            ⚠️ <i>Hanya untuk estimasi internal — bukan pengganti konsultan pajak bersertifikat.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()