import streamlit as st
import pandas as pd
import numpy as np
from svd_recommender import load_model, get_all_users

st.set_page_config(
    page_title="Kuliner Magelang",
    page_icon="🍜",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Lato', sans-serif !important;
    background-color: #FAF7F2 !important;
    color: #3B2A1A !important;
}
.stApp { background: #FAF7F2 !important; }
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #2C1F10 0%, #5A3A1E 55%, #96673A 100%);
    padding: 44px 60px 40px;
    position: relative; overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: rgba(166,124,82,0.12);
}
.hero-eyebrow {
    font-size: 10px; font-weight: 700;
    letter-spacing: 3.5px; text-transform: uppercase;
    color: #C9A882; margin-bottom: 12px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 54px; font-weight: 700;
    color: #F5EFE0; line-height: 1.05; margin: 0 0 10px;
}
.hero-title em { font-style: italic; color: #E8A87C; }
.hero-sub { font-size: 14px; color: #C9A882; font-weight: 300; letter-spacing: 0.3px; }

/* ── Nav tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #F5EFE0 !important;
    border-bottom: 1px solid #DDD0BA !important;
    padding: 0 60px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: #A67C52 !important;
    padding: 16px 24px !important;
    border-bottom: 3px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #3B2A1A !important;
    border-bottom: 3px solid #E8A87C !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 36px 60px !important;
    background: #FAF7F2 !important;
}

/* ── Sidebar boxes ── */
.stat-box {
    background: #F5EFE0; border-radius: 12px;
    padding: 20px 20px 14px; border: 1px solid #DDD0BA; margin-bottom: 18px;
}
.box-label {
    font-size: 9px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #A67C52; margin-bottom: 14px;
}
.stat-row {
    display: flex; justify-content: space-between;
    padding: 9px 0; border-bottom: 1px solid #DDD0BA; font-size: 13px;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #6B4C2A; }
.stat-val { font-weight: 700; color: #3B2A1A; }
.about-text { font-size: 12px; color: #6B4C2A; line-height: 1.85; }

/* ── Section headers ── */
.sect-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px; font-weight: 600; color: #3B2A1A; margin-bottom: 4px;
}
.sect-sub { font-size: 12px; color: #A67C52; margin-bottom: 10px; letter-spacing: 0.3px; }
.divider { width: 36px; height: 3px; background: #E8A87C; border-radius: 2px; margin-bottom: 26px; }

/* ── Form controls ── */
.stSelectbox > div > div {
    background: #F5EFE0 !important; border: 1.5px solid #DDD0BA !important;
    border-radius: 8px !important; color: #3B2A1A !important;
}
.stSelectbox label, .stTextInput label {
    font-size: 9px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: #A67C52 !important;
}
.stTextInput > div > div > input {
    background: #F5EFE0 !important; border: 1.5px solid #DDD0BA !important;
    border-radius: 8px !important; color: #3B2A1A !important; font-size: 13px !important;
}
.stButton > button {
    background: #3B2A1A !important; color: #F5EFE0 !important;
    border: none !important; border-radius: 8px !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    padding: 14px 28px !important; width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #6B4C2A !important; }

/* ── Ranking table ── */
.user-badge {
    display: inline-block; background: #3B2A1A; color: #F5EFE0;
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    padding: 4px 14px; border-radius: 20px; margin-bottom: 14px;
}
.rank-wrap {
    border-radius: 12px; overflow: hidden;
    border: 1px solid #DDD0BA;
    box-shadow: 0 6px 24px rgba(59,42,26,0.08);
}
.rank-header {
    display: grid; grid-template-columns: 52px 1fr 90px 90px;
    padding: 12px 20px; gap: 12px; background: #3B2A1A;
    font-size: 9px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #D4B896;
}
.rank-row {
    display: grid; grid-template-columns: 52px 1fr 90px 90px;
    padding: 15px 20px; gap: 12px;
    border-bottom: 1px solid #EEE5D5;
    background: #fff; align-items: center; cursor: pointer;
    transition: background 0.15s;
}
.rank-row:last-child { border-bottom: none; }
.rank-row:hover { background: #FDF8F0; }
.rank-num {
    font-family: 'Playfair Display', serif;
    font-size: 22px; font-weight: 700;
    color: #DDD0BA; text-align: center; line-height: 1;
}
.rank-num-top { color: #E8A87C !important; }
.rank-name { font-size: 13px; font-weight: 700; color: #3B2A1A; line-height: 1.35; }
.rank-score { text-align: center; font-size: 13px; font-weight: 700; color: #6B4C2A; }
.rank-stars { text-align: center; font-size: 13px; color: #D4920A; }

/* ── Detail card ── */
.detail-card {
    background: #F5EFE0; border-radius: 14px;
    border: 1px solid #DDD0BA; padding: 28px 30px; margin-top: 16px;
    box-shadow: 0 4px 20px rgba(59,42,26,0.06);
}
.detail-name {
    font-family: 'Playfair Display', serif;
    font-size: 22px; font-weight: 700; color: #3B2A1A; margin-bottom: 4px;
}
.detail-score-big {
    font-family: 'Playfair Display', serif;
    font-size: 48px; font-weight: 700; color: #E8A87C; line-height: 1;
}
.detail-label { font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #A67C52; margin-bottom: 6px; margin-top: 16px; }
.detail-val { font-size: 14px; color: #3B2A1A; font-weight: 700; }
.detail-stars { font-size: 20px; color: #D4920A; margin-top: 2px; }

/* ── Filter chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
.chip {
    background: #F5EFE0; border: 1.5px solid #DDD0BA;
    border-radius: 20px; padding: 5px 14px;
    font-size: 11px; font-weight: 700; color: #6B4C2A;
    cursor: pointer; letter-spacing: 0.5px;
}
.chip-active {
    background: #3B2A1A; border-color: #3B2A1A; color: #F5EFE0;
}

/* ── Chart area ── */
.chart-title {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 600; color: #3B2A1A; margin-bottom: 4px;
}
.chart-sub { font-size: 11px; color: #A67C52; margin-bottom: 20px; }

/* ── Empty ── */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 48px; opacity: 0.4; margin-bottom: 12px; }
.empty-text { font-family: 'Playfair Display', serif; font-size: 15px; color: #A67C52; }

/* ── Expander polish ── */
.streamlit-expanderHeader {
    background: #F5EFE0 !important; border-radius: 8px !important;
    font-weight: 700 !important; color: #3B2A1A !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_cached():
    return load_model()

@st.cache_data
def load_data():
    df_ui = pd.read_csv('user_item_matrix.csv')
    try:
        df_clean = pd.read_csv('data_clean.csv')
        # Normalize column names
        df_clean.columns = [c.strip().lower() for c in df_clean.columns]
        # Fix rating column
        if 'rating' in df_clean.columns:
            df_clean['rating'] = df_clean['rating'].astype(str).str.replace(',', '.').astype(float)
    except:
        df_clean = None
    return df_ui, df_clean

model = load_model_cached()
users = get_all_users()
df_ui, df_clean = load_data()

# ── Helper ─────────────────────────────────────────────────────────────────────
def make_stars(score):
    n = round(score)
    return "★" * n + "☆" * (5 - n)

def get_item_avg_rating():
    return df_ui.groupby('item')['rating'].mean().reset_index().rename(columns={'rating': 'avg_rating'})

def get_categories():
    if df_clean is not None:
        for col in ['kategori', 'category', 'tipe', 'jenis', 'type']:
            if col in df_clean.columns:
                cats = df_clean[col].dropna().unique().tolist()
                return sorted([str(c) for c in cats if str(c).strip()])
    return []

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Sistem Rekomendasi · SVD Collaborative Filtering</div>
    <div class="hero-title">Kuliner <em>Magelang</em></div>
    <div class="hero-sub">Temukan sajian terbaik yang sesuai seleramu
            </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Rekomendasi", "Eksplorasi Data", "Tentang Sistem"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REKOMENDASI
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_side, col_main = st.columns([1, 2.8], gap="large")

    # ── Sidebar
    with col_side:
        avg_df = get_item_avg_rating()
        total_items = len(avg_df)
        max_rating = avg_df['avg_rating'].max()
        mean_rating = avg_df['avg_rating'].mean()

        st.markdown(f"""
        <div class="stat-box">
            <div class="box-label">Statistik Dataset</div>
            <div class="stat-row"><span class="stat-key">Total Kuliner</span><span class="stat-val">{total_items}</span></div>
            <div class="stat-row"><span class="stat-key">Total User</span><span class="stat-val">{len(users)}</span></div>
            <div class="stat-row"><span class="stat-key">Rating Tertinggi</span><span class="stat-val">{max_rating:.1f} ⭐</span></div>
            <div class="stat-row"><span class="stat-key">Rata-rata Rating</span><span class="stat-val">{mean_rating:.2f} ⭐</span></div>
        </div>
        <div class="stat-box">
            <div class="box-label">Tentang</div>
            <div class="about-text">
                Sistem rekomendasi kuliner Magelang menggunakan algoritma <strong>SVD</strong>
                (<em>Singular Value Decomposition</em>) berbasis collaborative filtering
                untuk menghasilkan rekomendasi yang dipersonalisasi per pengguna.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Main
    with col_main:
        st.markdown('<div class="sect-title">Rekomendasi Personal</div>', unsafe_allow_html=True)
        st.markdown('<div class="sect-sub">Pilih user ID untuk melihat top-10 rekomendasi kuliner terbaik</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        user_id = st.selectbox("Pilih User ID", users, key="rec_user")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Tampilkan Rekomendasi"):
            recs = model.recommend(user_id, n=10)

            if recs:
                rows_html = ""
                for i, (item, pred) in enumerate(recs, 1):
                    top = " rank-num-top" if i <= 3 else ""
                    rows_html += f"""<div class="rank-row">
<div class="rank-num{top}">{i}</div>
<div class="rank-name">{item}</div>
<div class="rank-score">{pred:.2f}</div>
<div class="rank-stars">{make_stars(pred)}</div>
</div>"""

                st.markdown(f"""
<div class="user-badge">👤 {user_id}</div>
<div class="rank-wrap">
<div class="rank-header">
<div style="text-align:center">#</div>
<div>Nama Kuliner</div>
<div style="text-align:center">Prediksi</div>
<div style="text-align:center">Bintang</div>
</div>
{rows_html}
</div>""", unsafe_allow_html=True)

                # ── Detail expander
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("📊 Lihat Detail Kuliner"):
                    sel_item = st.selectbox(
                        "Pilih kuliner untuk detail",
                        [r[0] for r in recs],
                        key="detail_sel"
                    )
                    sel_pred = next(p for n, p in recs if n == sel_item)

                    # Get actual ratings for this item
                    item_ratings = df_ui[df_ui['item'] == sel_item]['rating']
                    actual_mean = item_ratings.mean() if len(item_ratings) > 0 else sel_pred
                    n_ratings = len(item_ratings)

                    # Extra info from data_clean if available
                    extra_info = {}
                    if df_clean is not None:
                        row = df_clean[df_clean['item'] == sel_item]
                        if len(row) > 0:
                            row = row.iloc[0]
                            for col in ['kategori', 'category', 'tipe', 'jenis', 'alamat', 'address', 'lokasi']:
                                if col in df_clean.columns and pd.notna(row.get(col)):
                                    extra_info[col] = row[col]

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="detail-card">
                            <div class="box-label">Prediksi Rating</div>
                            <div class="detail-score-big">{sel_pred:.2f}</div>
                            <div class="detail-stars">{make_stars(sel_pred)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="detail-card">
                            <div class="box-label">Rating Aktual</div>
                            <div class="detail-score-big">{actual_mean:.2f}</div>
                            <div class="detail-stars">{make_stars(actual_mean)}</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="detail-card">
                            <div class="box-label">Jumlah Rating</div>
                            <div class="detail-score-big">{n_ratings}</div>
                            <div class="detail-label">pengguna</div>
                        </div>""", unsafe_allow_html=True)

                    if extra_info:
                        extra_html = '<div class="detail-card" style="margin-top:12px">'
                        for k, v in extra_info.items():
                            extra_html += f'<div class="detail-label">{k.capitalize()}</div><div class="detail-val">{v}</div>'
                        extra_html += '</div>'
                        st.markdown(extra_html, unsafe_allow_html=True)

                    # Mini bar chart of rating distribution
                    if n_ratings > 0:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="chart-title">Distribusi Rating</div>', unsafe_allow_html=True)
                        dist = item_ratings.value_counts().sort_index()
                        st.bar_chart(dist, color="#A67C52")

            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🍽️</div>
                    <div class="empty-text">Tidak ada rekomendasi untuk user ini</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🍜</div>
                <div class="empty-text">Pilih user ID dan klik tombol untuk melihat rekomendasi</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EKSPLORASI DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sect-title">Eksplorasi Data Kuliner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sect-sub">Telusuri dan filter seluruh data kuliner Magelang</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    avg_df = get_item_avg_rating()
    avg_df = avg_df.sort_values('avg_rating', ascending=False).reset_index(drop=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search = st.text_input("Cari nama kuliner", placeholder="Contoh: soto, bakso, nasi...", key="search")
    with col_f2:
        sort_by = st.selectbox("Urutkan", ["Rating Tertinggi", "Rating Terendah", "Nama A-Z"], key="sort")

    # Apply filters
    filtered = avg_df.copy()
    if search:
        filtered = filtered[filtered['item'].str.lower().str.contains(search.lower(), na=False)]
    if sort_by == "Rating Terendah":
        filtered = filtered.sort_values('avg_rating', ascending=True)
    elif sort_by == "Nama A-Z":
        filtered = filtered.sort_values('item', ascending=True)

    st.markdown(f"<p style='font-size:12px;color:#A67C52;margin-bottom:16px'>Menampilkan <strong>{len(filtered)}</strong> dari {len(avg_df)} kuliner</p>", unsafe_allow_html=True)

    # Table
    if len(filtered) > 0:
        rows_html = ""
        for i, row in filtered.head(50).iterrows():
            rank_display = i + 1
            top = " rank-num-top" if rank_display <= 3 else ""
            rows_html += f"""<div class="rank-row">
<div class="rank-num{top}">{rank_display}</div>
<div class="rank-name">{row['item']}</div>
<div class="rank-score">{row['avg_rating']:.2f}</div>
<div class="rank-stars">{make_stars(row['avg_rating'])}</div>
</div>"""

        st.markdown(f"""
<div class="rank-wrap">
<div class="rank-header">
<div style="text-align:center">#</div>
<div>Nama Kuliner</div>
<div style="text-align:center">Avg Rating</div>
<div style="text-align:center">Bintang</div>
</div>
{rows_html}
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">Kuliner tidak ditemukan</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts
    st.markdown("<br><br>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown('<div class="chart-title">Distribusi Rating Semua Kuliner</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Persebaran nilai rata-rata rating per kuliner</div>', unsafe_allow_html=True)
        hist_data = pd.cut(avg_df['avg_rating'], bins=[1, 2, 3, 3.5, 4, 4.5, 5.01],
                           labels=['1-2', '2-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5'])
        hist_counts = hist_data.value_counts().sort_index()
        st.bar_chart(hist_counts, color="#A67C52")

    with ch2:
        st.markdown('<div class="chart-title">Top 10 Kuliner Terpopuler</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Berdasarkan rata-rata rating tertinggi</div>', unsafe_allow_html=True)
        top10 = avg_df.head(10).set_index('item')['avg_rating']
        st.bar_chart(top10, color="#6B4C2A")

    # Rating per user distribution
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Jumlah Rating per User</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-sub">Aktivitas rating masing-masing pengguna</div>', unsafe_allow_html=True)
    user_counts = df_ui.groupby('user_id').size().reset_index(name='jumlah_rating')
    user_counts = user_counts.set_index('user_id')
    st.bar_chart(user_counts, color="#96673A")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TENTANG SISTEM
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown('<div class="sect-title">Tentang Sistem</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-text" style="font-size:14px;line-height:2">
            Sistem rekomendasi ini dibangun sebagai bagian dari penelitian skripsi
            dengan fokus pada penerapan <strong>Collaborative Filtering</strong> berbasis
            <strong>SVD (Singular Value Decomposition)</strong> untuk domain kuliner lokal Magelang.
            <br><br>
            <strong>Alur Sistem:</strong><br>
            Data rating pengguna dikumpulkan dan direpresentasikan sebagai matriks user-item.
            Matriks ini kemudian didekomposisi menggunakan SVD untuk mengekstrak faktor laten
            yang merepresentasikan preferensi pengguna dan karakteristik kuliner.
            Prediksi rating dihasilkan dari rekonstruksi matriks hasil dekomposisi.
        </div>
        """, unsafe_allow_html=True)

    with c2:
        avg_df = get_item_avg_rating()
        st.markdown(f"""
        <div class="stat-box">
            <div class="box-label">Spesifikasi Model</div>
            <div class="stat-row"><span class="stat-key">Algoritma</span><span class="stat-val">SVD</span></div>
            <div class="stat-row"><span class="stat-key">Metode</span><span class="stat-val">Collaborative Filtering</span></div>
            <div class="stat-row"><span class="stat-key">Faktor Laten</span><span class="stat-val">15</span></div>
            <div class="stat-row"><span class="stat-key">Total Item</span><span class="stat-val">{len(avg_df)}</span></div>
            <div class="stat-row"><span class="stat-key">Total User</span><span class="stat-val">{len(users)}</span></div>
            <div class="stat-row"><span class="stat-key">Top-N</span><span class="stat-val">10</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-box" style="max-width:600px">
        <div class="box-label">Teknologi yang Digunakan</div>
        <div class="stat-row"><span class="stat-key">Framework UI</span><span class="stat-val">Streamlit</span></div>
        <div class="stat-row"><span class="stat-key">Bahasa</span><span class="stat-val">Python 3.14</span></div>
        <div class="stat-row"><span class="stat-key">Library Utama</span><span class="stat-val">NumPy · SciPy · Pandas</span></div>
        <div class="stat-row"><span class="stat-key">SVD Engine</span><span class="stat-val">scipy.sparse.linalg.svds</span></div>
    </div>
    """, unsafe_allow_html=True)