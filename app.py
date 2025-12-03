import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# --- Konfigurasi Halaman ---
st.set_page_config(layout="wide", page_title="Business Clustering in Kota Bekasi")

# --- 1. Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    try:
        # Menggunakan os untuk mendeteksi lokasi file secara absolut
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "dataset.csv")
        
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("dataset.csv tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py")
        st.stop()

    dataset = data.copy()

    # Membersihkan kolom yang tidak perlu
    cols_to_drop = [
        "Kode Pos", "Sektor Institusi", "Unnamed: 17", "Sumber Profiling",
        "Catatan Profiling", "IDSBR.1", "Nama Usaha.1", "Kegiatan Usaha",
        "Kategori.1", "KBLI.1"
    ]
    dataset = dataset.drop(columns=[col for col in cols_to_drop if col in dataset.columns])

    # Konversi tipe data
    if "IDSBR" in dataset.columns:
        dataset["IDSBR"] = pd.to_numeric(dataset["IDSBR"], errors="coerce")
    if "KBLI" in dataset.columns:
        dataset["KBLI"] = pd.to_numeric(dataset["KBLI"], errors="coerce")

    # Rename kolom
    if "Kelurahan/Desa" in dataset.columns:
        dataset.rename(columns={"Kelurahan/Desa": "NAMOBJ"}, inplace=True)

    # Filter status usaha
    values_to_remove = ["Duplikat", "Tidak Ditemukan", "Tutup", "Tutup Sementara"]
    if "Keberadaan Usaha/Perusahaan" in dataset.columns:
        dataset = dataset[~dataset["Keberadaan Usaha/Perusahaan"].isin(values_to_remove)]

    return dataset


@st.cache_data
def load_geojson():
    try:
        # Menggunakan os untuk mendeteksi lokasi shapefile secara absolut
        current_dir = os.path.dirname(os.path.abspath(__file__))
        shp_path = os.path.join(current_dir, "ADMINISTRASIDESA_AR_25K.shp")
        
        prov = gpd.read_file(shp_path)
    except FileNotFoundError:
        st.error("ADMINISTRASIDESA_AR_25K.shp tidak ditemukan. Letakkan berkas shapefile di folder aplikasi.")
        st.stop()
    return prov


def prepare_bekasi_data_geo(dataset):
    # Filter khusus Kota Bekasi
    bekasi_data = dataset[dataset["Kabupaten/Kota"] == "[75] BEKASI"].copy()

    bekasi_data["Latitude"] = pd.to_numeric(bekasi_data["Latitude"], errors="coerce")
    bekasi_data["Longitude"] = pd.to_numeric(bekasi_data["Longitude"], errors="coerce")

    bekasi_data = bekasi_data.dropna(subset=["Latitude", "Longitude"])

    bekasi_data_geo = gpd.GeoDataFrame(
        bekasi_data,
        geometry=gpd.points_from_xy(bekasi_data["Longitude"], bekasi_data["Latitude"]),
        crs="EPSG:4326"
    )
    return bekasi_data_geo


# PERBAIKAN PENTING: Menambahkan underscore (_) pada nama argumen
# agar Streamlit tidak mencoba melakukan hashing pada GeoDataFrame
@st.cache_data
def perform_clustering(_bekasi_data_geo):
    # Buat copy agar tidak memodifikasi dataframe asli di luar fungsi secara langsung
    df_geo = _bekasi_data_geo.copy()
    
    coordinates = df_geo[["Latitude", "Longitude"]]

    scaler = StandardScaler()
    coordinates_scaled = scaler.fit_transform(coordinates)

    model = AgglomerativeClustering(n_clusters=5)
    clusters = model.fit_predict(coordinates_scaled)

    df_geo["cluster"] = clusters

    score = silhouette_score(coordinates_scaled, clusters)

    return df_geo, score


# --- Streamlit UI Layout ---
st.title("Analisis Klaster Usaha di Kota Bekasi")
st.markdown("---")

# Load data
with st.spinner("Memuat data..."):
    dataset = load_data()
    prov_gdf = load_geojson()
    bekasi_data_geo = prepare_bekasi_data_geo(dataset)

if bekasi_data_geo.empty:
    st.warning("Tidak ada data usaha valid di Kota Bekasi.")
    st.stop()

# Proses Clustering
with st.spinner("Melakukan clustering..."):
    # Panggil fungsi dengan argumen biasa (tanpa underscore saat pemanggilan)
    bekasi_data_geo_result, score = perform_clustering(bekasi_data_geo)

# Tampilkan Hasil
st.subheader("Hasil Clustering")
col1, col2 = st.columns([3, 1])

with col1:
    st.write(f"**Skor Silhouette:** {score:.3f}")
    
    # Visualisasi Peta
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot Peta Dasar (Shapefile)
    prov_gdf.plot(ax=ax, color="whitesmoke", edgecolor="lightgray")
    
    # Plot Titik Usaha Berdasarkan Cluster
    bekasi_data_geo_result.plot(column="cluster", ax=ax, legend=True, cmap='viridis', markersize=15, alpha=0.7)
    
    ax.set_title("Peta Sebaran Klaster Usaha")
    ax.set_axis_off()
    st.pyplot(fig)

with col2:
    st.subheader("Statistik Klaster")
    # Menampilkan ringkasan sederhana
    cluster_counts = bekasi_data_geo_result["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

st.markdown("---")
st.subheader("Ringkasan Karakteristik Bisnis per Klaster")

# Loop untuk menampilkan detail setiap cluster
unique_clusters = sorted(bekasi_data_geo_result["cluster"].unique())
cols = st.columns(len(unique_clusters))

for i, cluster in enumerate(unique_clusters):
    with cols[i]:
        st.markdown(f"#### Klaster {cluster}")
        data_c = bekasi_data_geo_result[bekasi_data_geo_result["cluster"] == cluster]
        
        st.info(f"Jumlah Usaha: {len(data_c)}")
        
        if "Kategori" in data_c.columns:
            st.caption("**Kategori Dominan:**")
            top_cat = data_c["Kategori"].value_counts().head(3)
            for cat, count in top_cat.items():
                st.write(f"- {cat} ({count})")
        
        if "Bentuk Badan Hukum/Usaha" in data_c.columns:
            st.caption("**Bentuk Usaha:**")
            top_legal = data_c["Bentuk Badan Hukum/Usaha"].value_counts().head(3)
            for leg, count in top_legal.items():
                st.write(f"- {leg} ({count})")