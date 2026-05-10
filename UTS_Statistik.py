import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os # Tambahan modul OS untuk mengecek file lokal

# Untuk analisis statistik lanjutan
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ================= KONFIGURASI HALAMAN =================
st.set_page_config(page_title="MMM Dashboard - UTS Statistik", page_icon="📊", layout="wide")

# ================= FUNGSI PEMBERSIH DATA (CACHE) =================
@st.cache_data
def load_and_clean_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    large_num_cols = [
        'Sales_Units', 'Sales_Value', 'MRP', 'Net_Price', 'Trade_Spend',
        'TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions', 
        'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership', 'TDP'
    ]
    decimal_cols = [
        'FB_Banner_Content_Score', 'IG_Banner_Content_Score', 
        'Weighted_Distribution', 'Numeric_Distribution', 'NOS', 'CPI', 
        'GDP_Growth', 'Festival_Index', 'Rainfall_Index'
    ]

    for col in large_num_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in decimal_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if all(c in df.columns for c in ['YouTube_Impressions', 'Facebook_Impressions', 'Instagram_Impressions']):
        df['Total_Digital_Imp'] = df['YouTube_Impressions'] + df['Facebook_Impressions'] + df['Instagram_Impressions']
    if all(c in df.columns for c in ['TV_Impressions', 'Print_Readership', 'Radio_Listenership']):
        df['Total_Trad_Imp'] = df['TV_Impressions'] + df['Print_Readership'] + df['Radio_Listenership']

    return df

def format_number(num):
    if pd.isna(num): return "0"
    if num >= 1e9: return f"{num/1e9:.2f} B"
    elif num >= 1e6: return f"{num/1e6:.2f} M"
    elif num >= 1e3: return f"{num/1e3:.2f} K"
    else: return f"{num:,.2f}"

# ================= HEADER & IDENTITAS UTAMA =================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://petra.ac.id/img/logo-pcu.4d2cad68.png", width=120)
with col_title:
    st.title("Dashboard Visualisasi & Analisis Data")
    st.markdown("**UTS Statistik** | **Nama:** Jonathan Deven | **NIM:** C23250006")

st.markdown("---")

# ================= SIDEBAR (KHUSUS KONTROL/FILTER) =================
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload dataset (CSV)", type=['csv'])

# ================= KONTEN UTAMA (LOGIKA DEFAULT DATASET) =================
# Tentukan path file default menggunakan raw string (r"") agar backslash terbaca benar
DEFAULT_DATA = "synthetic_mmm_weekly_india.csv"

# Logika penentuan dataset mana yang dipakai
df = None
if uploaded_file is not None:
    # 1. Prioritas utama: File dari user
    df = load_and_clean_data(uploaded_file)
    st.sidebar.success("✅ Menggunakan dataset yang diunggah.")
elif os.path.exists(DEFAULT_DATA):
    # 2. Prioritas kedua: File default lokal
    df = load_and_clean_data(DEFAULT_DATA)
    st.sidebar.info("ℹ️ Menggunakan dataset synthetic_mmm_weekly_india.csv")
else:
    # 3. Jika tidak ada keduanya
    st.markdown("Platform analitik interaktif berbasis data historis mingguan untuk mengevaluasi ROI pemasaran, efektivitas promosi, dan membedah dinamika penjualan melalui pemodelan statistik tingkat lanjut.")
    st.warning("⚠️ Dataset default tidak ditemukan. Silakan unggah file CSV di *sidebar* kiri untuk memuat dashboard UTS MMM Anda.")
    st.stop() # Hentikan eksekusi script di sini jika tidak ada data

# Lanjutkan render dashboard JIKA dataframe (df) sudah ada
if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Global Filters")
    selected_brand = st.sidebar.multiselect("Pilih Brand", options=df['Brand'].dropna().unique(), default=df['Brand'].dropna().unique())
    selected_geo = st.sidebar.multiselect("Pilih Wilayah (Geo)", options=df['Geo'].dropna().unique(), default=df['Geo'].dropna().unique())
    
    filtered_df = df[(df['Brand'].isin(selected_brand)) & (df['Geo'].isin(selected_geo))]
    
    if filtered_df.empty:
        st.warning("Data kosong dengan filter saat ini. Silakan sesuaikan filter Anda.")
        st.stop()

    if not HAS_STATSMODELS:
        st.warning("⚠️ Pustaka `statsmodels` tidak terdeteksi. Beberapa analisis di Tab 5 tidak akan berjalan. Jalankan `pip install statsmodels` di terminal Anda.")

    # List Variabel Numerik & Kategorik untuk Pilihan Dropdown
    num_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_cols = [c for c in num_cols if set(filtered_df[c].dropna().unique()).issubset({0, 1})]

    # ================= TABS =================
    t1, t2, t3, t4, t5 = st.tabs([
        "Executive Summary", 
        "Sales & Geo", 
        "Media & Promo", 
        "Distribusi & Makro",
        "Analisis Statistik Lanjut"
    ])

    # ---------------- TAB 1: EXECUTIVE SUMMARY ----------------
    with t1:
        total_sales_val = filtered_df['Sales_Value'].sum()
        total_sales_unit = filtered_df['Sales_Units'].sum()
        total_trade_spend = filtered_df['Trade_Spend'].sum()
        avg_price = filtered_df['Net_Price'].mean()
        est_roi = total_sales_val / total_trade_spend if total_trade_spend > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Sales Value", f"₹ {format_number(total_sales_val)}")
        col2.metric("Total Sales Units", format_number(total_sales_unit))
        col3.metric("Total Trade Spend", f"₹ {format_number(total_trade_spend)}")
        col4.metric("Avg Net Price", f"₹ {avg_price:,.2f}")
        col5.metric("Est. Trade ROI", f"{est_roi:.2f}x")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            df_brand = filtered_df.groupby('Brand')['Sales_Value'].sum().reset_index()
            fig_pie = px.pie(df_brand, names='Brand', values='Sales_Value', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel, title="Market Share by Brand")
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            df_sku = filtered_df.groupby(['Brand', 'SKU'])['Sales_Value'].sum().reset_index().sort_values(by='Sales_Value', ascending=False).head(10)
            fig_bar_sku = px.bar(df_sku, x='Sales_Value', y='SKU', color='Brand', orientation='h', text_auto='.2s', title="Top Performing SKUs")
            fig_bar_sku.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_bar_sku, use_container_width=True)

        st.markdown("---")
        df_tree = filtered_df.groupby(['Brand', 'SKU'])[['Sales_Value', 'Sales_Units']].sum().reset_index()
        fig_tree = px.treemap(df_tree, path=[px.Constant("All Brands"), 'Brand', 'SKU'], values='Sales_Value', color='Sales_Units', color_continuous_scale='Blues', title="Hierarki Kontribusi Penjualan (Treemap)")
        st.plotly_chart(fig_tree, use_container_width=True)

    # ---------------- TAB 2: SALES & GEO ----------------
    with t2:
        df_trend = filtered_df.groupby('Date')[['Sales_Value', 'Sales_Units']].sum().reset_index()
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(go.Bar(x=df_trend['Date'], y=df_trend['Sales_Units'], name="Sales Units", opacity=0.6, marker_color='#3498db'), secondary_y=False)
        fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Sales_Value'], name="Sales Value (₹)", mode='lines', line=dict(color='#e74c3c', width=3)), secondary_y=True)
        fig_trend.update_layout(title="Tren Penjualan Mingguan Nasional")
        st.plotly_chart(fig_trend, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            df_geo = filtered_df.groupby('Geo')['Sales_Value'].sum().reset_index().sort_values('Sales_Value', ascending=False)
            fig_geo = px.bar(df_geo, x='Geo', y='Sales_Value', color='Geo', text_auto='.2s', title="Kontribusi Penjualan per Wilayah")
            st.plotly_chart(fig_geo, use_container_width=True)
        with c2:
            fig_box = px.box(filtered_df, x='Geo', y='Sales_Units', color='Brand', title="Boxplot: Sebaran & Outlier Volume Penjualan")
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        fig_bubble = px.scatter(filtered_df, x="Net_Price", y="Sales_Units", size="Sales_Value", color="Geo", hover_name="Brand", log_x=True, size_max=40, title="Bubble Chart: Net Price vs Sales Units per Wilayah (Ukuran Bubble = Sales Value)")
        st.plotly_chart(fig_bubble, use_container_width=True)

    # ---------------- TAB 3: MEDIA & PROMO ----------------
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            media_sums = filtered_df[['TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions', 'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership']].sum().reset_index()
            media_sums.columns = ['Channel', 'Impressions']
            fig_media = px.bar(media_sums, x='Channel', y='Impressions', color='Channel', text_auto='.2s', title="Total Media Impressions")
            st.plotly_chart(fig_media, use_container_width=True)
        with c2:
            fig_spend = px.scatter(filtered_df, x='Trade_Spend', y='Sales_Value', color='TPR_Flag', size='Sales_Units', hover_data=['Brand', 'Geo'], title="Trade Spend vs Sales Value (Warna: TPR Aktif)")
            st.plotly_chart(fig_spend, use_container_width=True)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            df_radar = filtered_df.groupby('Brand')[['TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions', 'Instagram_Impressions']].mean().reset_index()
            fig_radar = go.Figure()
            for i, row in df_radar.iterrows():
                fig_radar.add_trace(go.Scatterpolar(r=row[1:].values, theta=row[1:].index, fill='toself', name=row['Brand']))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Radar Chart: Bauran Rata-Rata Media per Brand")
            st.plotly_chart(fig_radar, use_container_width=True)
        with c4:
            df_promo_time = filtered_df.groupby('Date')[['Trade_Spend']].sum().reset_index()
            fig_area = px.area(df_promo_time, x='Date', y='Trade_Spend', title="Area Chart: Intensitas Pengeluaran Promo (Trade Spend) Sepanjang Waktu", color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig_area, use_container_width=True)

    # ---------------- TAB 4: DISTRIBUSI & MAKRO ----------------
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            df_dist = filtered_df.groupby('Date')[['Weighted_Distribution', 'Numeric_Distribution']].mean().reset_index()
            fig_dist = px.line(df_dist, x='Date', y=['Weighted_Distribution', 'Numeric_Distribution'], title="Tren Distribusi Nasional")
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            fig_dist_scatter = px.scatter(filtered_df, x="Numeric_Distribution", y="Weighted_Distribution", color="Brand", title="Korelasi: Numeric vs Weighted Distribution", trendline="ols")
            st.plotly_chart(fig_dist_scatter, use_container_width=True)

        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            df_mac = filtered_df.groupby('Date')[['CPI', 'GDP_Growth']].mean().reset_index()
            fig_mac = make_subplots(specs=[[{"secondary_y": True}]])
            fig_mac.add_trace(go.Scatter(x=df_mac['Date'], y=df_mac['CPI'], name='CPI (Inflasi)'), secondary_y=False)
            fig_mac.add_trace(go.Scatter(x=df_mac['Date'], y=df_mac['GDP_Growth'], name='GDP Growth'), secondary_y=True)
            fig_mac.update_layout(title="Indikator Makro: CPI & GDP Growth")
            st.plotly_chart(fig_mac, use_container_width=True)
        with c4:
            df_ext = filtered_df.groupby('Date')[['Sales_Units', 'Festival_Index', 'Rainfall_Index']].mean().reset_index()
            fig_ext = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ext.add_trace(go.Bar(x=df_ext['Date'], y=df_ext['Festival_Index'], name="Festival Index", opacity=0.3, marker_color='orange'), secondary_y=False)
            fig_ext.add_trace(go.Bar(x=df_ext['Date'], y=df_ext['Rainfall_Index'], name="Rainfall Index", opacity=0.3, marker_color='lightblue'), secondary_y=False)
            fig_ext.add_trace(go.Scatter(x=df_ext['Date'], y=df_ext['Sales_Units'], name="Sales Units", mode='lines', line=dict(color='#2ecc71')), secondary_y=True)
            fig_ext.update_layout(title="Overlay Penjualan dengan Festival & Curah Hujan")
            st.plotly_chart(fig_ext, use_container_width=True)

    # ---------------- TAB 5: ANALISIS STATISTIK LANJUT ----------------
    with t5:
        st.header("Pemodelan & Uji Hipotesis Statistik")
        
        # 5.1 Statistik Deskriptif & Korelasi (Lebar Penuh)
        with st.expander("1. Statistik Deskriptif & Heatmap Korelasi", expanded=True):
            st.markdown("**Ringkasan Metrik Numerik Utama (Deskriptif)**")
            st.dataframe(filtered_df[num_cols].describe().T.style.format("{:,.2f}"), use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Matriks Korelasi (Pearson)**")
            default_corr = ['Sales_Units', 'Sales_Value', 'Net_Price', 'Trade_Spend', 'Weighted_Distribution']
            default_corr = [c for c in default_corr if c in num_cols]
            corr_vars = st.multiselect("Pilih Variabel untuk Korelasi", num_cols, default=default_corr)
            if len(corr_vars) > 1:
                corr_matrix = filtered_df[corr_vars].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Pilih setidaknya 2 variabel untuk melihat matriks korelasi.")

        if HAS_STATSMODELS and len(filtered_df) > 10:
            # 5.2 Regresi Linear Berganda (OLS)
            with st.expander("2. Regresi Linear Berganda (Multivariate OLS)", expanded=False):
                c_ols_1, c_ols_2 = st.columns([1, 2])
                with c_ols_1:
                    st.markdown("**Parameter Model**")
                    default_y_ols = 'Sales_Units' if 'Sales_Units' in num_cols else num_cols[0]
                    y_ols = st.selectbox("Variabel Target (Y)", num_cols, index=num_cols.index(default_y_ols), key="ols_y")
                    
                    default_x_ols = [c for c in ['Net_Price', 'Weighted_Distribution', 'Festival_Index'] if c in num_cols]
                    x_ols = st.multiselect("Variabel Prediktor (X)", num_cols, default=default_x_ols, key="ols_x")
                
                with c_ols_2:
                    if y_ols and x_ols:
                        try:
                            ols_data = filtered_df[[y_ols] + x_ols].dropna()
                            X = sm.add_constant(ols_data[x_ols])
                            y = ols_data[y_ols]
                            model = sm.OLS(y, X).fit()
                            
                            summary_df = pd.DataFrame({
                                "Koefisien": model.params,
                                "Std Error": model.bse,
                                "t-value": model.tvalues,
                                "P>|t|": model.pvalues
                            })
                            
                            st.markdown(f"**R-Squared:** {model.rsquared:.4f}  |  **F-Statistic Prob:** {model.f_pvalue:.4e}")
                            st.dataframe(summary_df.style.format("{:.4f}").highlight_between(left=0, right=0.05, subset=['P>|t|'], color='#c8e6c9'), use_container_width=True)
                            st.caption("P-value < 0.05 (hijau) menandakan variabel signifikan.")
                        except Exception as e:
                            st.error(f"Error Model OLS: {e}")
                    else:
                        st.warning("Pilih Variabel Target (Y) dan minimal satu Prediktor (X).")

            # 5.3 Mixed-Effects Model (Hierarchical Regression)
            with st.expander("3. Mixed-Effects Model (Random Intercept)", expanded=False):
                c_me_1, c_me_2 = st.columns([1, 2])
                with c_me_1:
                    st.markdown("**Parameter Model**")
                    y_me = st.selectbox("Variabel Target (Y)", num_cols, index=num_cols.index(default_y_ols), key="me_y")
                    x_me = st.selectbox("Variabel Prediktor (Fixed Effect)", num_cols, index=num_cols.index('Net_Price') if 'Net_Price' in num_cols else 0, key="me_x")
                    
                    group_options = [c for c in ['Geo', 'Brand', 'SKU'] if c in cat_cols]
                    group_me = st.selectbox("Variabel Grouping (Random Effect)", group_options, key="me_group") if group_options else None
                
                with c_me_2:
                    if y_me and x_me and group_me:
                        try:
                            me_data = filtered_df[[y_me, x_me, group_me]].dropna()
                            formula = f"{y_me} ~ {x_me}"
                            md = smf.mixedlm(formula, me_data, groups=me_data[group_me])
                            mdf = md.fit()
                            st.text(mdf.summary().as_text())
                        except Exception as e:
                            st.error(f"Error MixedLM: Matriks singular atau data tidak mencukupi per grup. Detail: {e}")
                    else:
                        st.warning("Pastikan Anda memiliki variabel kategori untuk Grouping.")

            # 5.4 Causal Inference (Average Treatment Effect)
            with st.expander("4. Causal Inference: Average Treatment Effect", expanded=False):
                c_ci_1, c_ci_2 = st.columns([1, 2])
                with c_ci_1:
                    st.markdown("**Parameter Model**")
                    y_ci = st.selectbox("Variabel Target (Y)", num_cols, index=num_cols.index(default_y_ols), key="ci_y")
                    
                    if binary_cols:
                        default_treat = 'TPR_Flag' if 'TPR_Flag' in binary_cols else binary_cols[0]
                        treat_ci = st.selectbox("Treatment Biner (0/1)", binary_cols, index=binary_cols.index(default_treat), key="ci_treat")
                    else:
                        treat_ci = None
                        st.error("Tidak ada kolom bernilai Biner (0/1) murni di dataset untuk dijadikan Treatment.")
                    
                    default_conf = [c for c in ['Weighted_Distribution', 'CPI', 'Festival_Index'] if c in num_cols]
                    conf_ci = st.multiselect("Variabel Pengontrol (Confounders)", num_cols, default=default_conf, key="ci_conf")
                
                with c_ci_2:
                    if y_ci and treat_ci:
                        try:
                            cols_to_keep = [y_ci, treat_ci] + conf_ci
                            ci_data = filtered_df[cols_to_keep].dropna()
                            
                            # 1. Naive ATE
                            mean_treated = ci_data[ci_data[treat_ci] == 1][y_ci].mean()
                            mean_control = ci_data[ci_data[treat_ci] == 0][y_ci].mean()
                            naive_ate = mean_treated - mean_control
                            
                            # 2. Adjusted ATE (OLS)
                            X_ci = sm.add_constant(ci_data[[treat_ci] + conf_ci])
                            ci_model = sm.OLS(ci_data[y_ci], X_ci).fit()
                            adjusted_ate = ci_model.params[treat_ci]
                            p_val_ci = ci_model.pvalues[treat_ci]
                            
                            k1, k2, k3 = st.columns(3)
                            k1.metric("Naive ATE (Tanpa Kontrol)", f"{naive_ate:+,.2f}")
                            k2.metric("Adjusted ATE (Terkontrol)", f"{adjusted_ate:+,.2f}")
                            k3.metric("P-Value (Signifikansi)", f"{p_val_ci:.4f}")
                            
                            st.markdown(f"**Interpretasi:** Jika diterapkan Treatment `{treat_ci}`, variabel `{y_ci}` diproyeksikan akan berubah sebesar **{adjusted_ate:+.2f}**, setelah mengontrol bias dari efek pengganggu ({', '.join(conf_ci) if conf_ci else 'Tidak ada'}).")
                        except Exception as e:
                            st.error(f"Gagal memproses Causal Inference. Detail: {e}")
