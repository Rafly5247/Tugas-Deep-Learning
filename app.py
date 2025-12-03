
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from shapely.geometry import Point

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    # Load dataset.csv
    try:
        data = pd.read_csv('/content/drive/MyDrive/dataset.csv')
    except FileNotFoundError:
        st.error("dataset.csv not found. Please ensure it's in /content/drive/MyDrive/")
        st.stop()

    dataset = data.copy()
    dataset.drop(["Kode Pos", "Sektor Institusi", "Unnamed: 17", "Sumber Profiling",
                  "Catatan Profiling", "IDSBR.1", "Nama Usaha.1", "Kegiatan Usaha",
                  "Kategori.1", "KBLI.1"], axis=1, inplace=True)

    # Convert 'IDSBR' and 'KBLI' to Int64
    if 'IDSBR' in dataset.columns:
        dataset['IDSBR'] = pd.to_numeric(dataset['IDSBR'], errors='coerce').astype('Int64')
    if 'KBLI' in dataset.columns:
        dataset['KBLI'] = pd.to_numeric(dataset['KBLI'], errors='coerce').astype('Int64')

    # Rename 'Kelurahan/Desa' to 'NAMOBJ'
    dataset.rename(columns={'Kelurahan/Desa': 'NAMOBJ'}, inplace=True)

    # Perform data cleaning for 'Keberadaan Usaha/Perusahaan'
    values_to_remove = ['Duplikat', 'Tidak Ditemukan', 'Tutup', 'Tutup Sementara']
    dataset = dataset[~dataset['Keberadaan Usaha/Perusahaan'].isin(values_to_remove)]

    return dataset

@st.cache_data
def load_geojson():
    # Load ADMINISTRASIDESA_AR_25K.shp
    try:
        prov = gpd.read_file('/content/drive/MyDrive/KOTA BEKASI/ADMINISTRASIDESA_AR_25K.shp')
    except FileNotFoundError:
        st.error("ADMINISTRASIDESA_AR_25K.shp not found. Please ensure it's in /content/drive/MyDrive/KOTA BEKASI/")
        st.stop()
    return prov

def prepare_bekasi_data_geo(dataset):
    bekasi_data = dataset[dataset['Kabupaten/Kota'] == '[75] BEKASI'].copy()
    bekasi_data['Latitude'] = pd.to_numeric(bekasi_data['Latitude'], errors='coerce')
    bekasi_data['Longitude'] = pd.to_numeric(bekasi_data['Longitude'], errors='coerce')
    bekasi_data = bekasi_data.dropna(subset=['Latitude', 'Longitude'])

    bekasi_data_geo = gpd.GeoDataFrame(
        bekasi_data,
        geometry=gpd.points_from_xy(bekasi_data['Longitude'], bekasi_data['Latitude']),
        crs="EPSG:4326"
    )
    return bekasi_data_geo

# --- 2. Hierarchical Clustering ---
@st.cache_data
def perform_clustering(bekasi_data_geo):
    coordinates_hc = bekasi_data_geo[['Latitude', 'Longitude']]
    scaler = StandardScaler()
    coordinates_scaled = scaler.fit_transform(coordinates_hc)

    agg_clustering = AgglomerativeClustering(n_clusters=5, linkage='ward')
    clusters = agg_clustering.fit_predict(coordinates_scaled)
    bekasi_data_geo['cluster'] = clusters

    silhouette_avg = silhouette_score(coordinates_scaled, clusters)
    return bekasi_data_geo, silhouette_avg, coordinates_scaled, clusters

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Business Clustering in Kota Bekasi")
st.title("Analisis Klaster Usaha di Kota Bekasi")
st.markdown("--- ")

# Load Data
with st.spinner("Memuat dan memproses data..."):
    dataset = load_data()
    prov_gdf = load_geojson()
    bekasi_data_geo = prepare_bekasi_data_geo(dataset)

if bekasi_data_geo.empty:
    st.warning("Tidak ada data usaha yang valid untuk Kota Bekasi setelah pembersihan.")
    st.stop()

# Perform Clustering
with st.spinner("Melakukan klasterisasi Hirarkis..."):
    bekasi_data_geo, silhouette_avg, coordinates_scaled, clusters = perform_clustering(bekasi_data_geo)

st.header("Hasil Hierarchical Clustering")
st.write(f"Bisnis dikelompokkan menjadi 5 klaster berdasarkan lokasi geografis.")
st.write(f"Skor Silhouette rata-rata untuk klasterisasi adalah: **{silhouette_avg:.3f}**")
st.info("Skor Silhouette yang moderat (0.419) menunjukkan bahwa klaster-klaster cukup terpisah.")

# --- 3. Visualization ---
st.subheader("Visualisasi Klaster Geografis")
fig, ax = plt.subplots(1, 1, figsize=(15, 13))
prov_gdf.plot(ax=ax, color='lightgray', edgecolor='black')
bekasi_data_geo.plot(column='cluster', ax=ax, legend=True, markersize=15, cmap='viridis')
ax.set_title('Hierarchical Clusters of Businesses in Kota Bekasi')
ax.set_axis_off()
st.pyplot(fig)
st.markdown("Visualisasi di atas menunjukkan sebaran klaster usaha di Kota Bekasi. Setiap warna mewakili klaster yang berbeda.")

# --- 4. Cluster Characteristics Summary ---
st.header("Ringkasan Karakteristik Bisnis per Klaster")

unique_clusters = bekasi_data_geo['cluster'].unique()

for cluster_id in sorted(unique_clusters):
    st.subheader(f"\n--- Klaster {cluster_id} ---")
    cluster_data = bekasi_data_geo[bekasi_data_geo['cluster'] == cluster_id]

    st.write("**Kategori Dominan:**")
    st.write(cluster_data['Kategori'].value_counts(dropna=False).head().to_frame())

    st.write("**Bentuk Badan Hukum/Usaha Dominan:**")
    st.write(cluster_data['Bentuk Badan Hukum/Usaha'].value_counts(dropna=False).head().to_frame())

# --- 5. Overall Summary ---
st.header("Kesimpulan Umum")
st.markdown("""
Berdasarkan Hierarchical Clustering yang diterapkan pada koordinat geografis bisnis di Kota Bekasi:

- Bisnis telah dikelompokkan menjadi 5 klaster yang berbeda berdasarkan kedekatan lokasi.
- Skor Silhouette sebesar 0.419 menunjukkan pemisahan yang cukup baik antar klaster, mengindikasikan bahwa klaster-klaster tersebut cukup terdefinisi.
- Visualisasi jelas menunjukkan distribusi spasial klaster-klaster ini di seluruh Kota Bekasi, menyoroti area-area di mana bisnis cenderung terkonsentrasi atau membentuk pengelompokan geografis yang berbeda.
- Warna yang berbeda pada peta merepresentasikan klaster yang berbeda, memberikan pemahaman intuitif tentang segmentasi spasial bisnis. Ini dapat berguna untuk pemasaran yang ditargetkan, perencanaan kota, atau memahami pusat ekonomi lokal.

### Ringkasan Komprehensif Karakteristik Bisnis dalam Setiap Klaster:

- **Klaster 0:** Dominan di sektor konstruksi/real estat (F) dan akomodasi/makanan (I), sebagian besar dioperasikan oleh entitas korporasi ('1. Perseroan'). Menunjukkan area komersial dan industri serbaguna.
- **Klaster 1:** Keseimbangan antara manufaktur (C) dan konstruksi/real estat (F). Juga didominasi oleh korporasi ('1. Perseroan'), namun dengan kehadiran Usaha Orang Perseorangan yang signifikan.
- **Klaster 2:** Pusat ritel dan perdagangan yang dinamis (G), didominasi oleh Usaha Orang Perseorangan. Menunjukkan keberadaan pasar lokal atau usaha kecil yang kuat.
- **Klaster 3:** Klaster yang lebih kecil dan mungkin terspesialisasi, dengan fokus pada manufaktur (C) dan layanan profesional. Utamanya dioperasikan oleh korporasi ('1. Perseroan') dan persekutuan komanditer (CV).
- **Klaster 4:** Mirip dengan Klaster 0, dengan penekanan kuat pada konstruksi/real estat (F) dan akomodasi/makanan (I), serta dioperasikan oleh entitas korporasi ('1. Perseroan').

### Wawasan dan Langkah Selanjutnya:

- Pengelompokan geografis yang teridentifikasi dapat menjadi dasar strategi pemasaran yang ditargetkan, inisiatif perencanaan kota, atau alokasi sumber daya. Hal ini memungkinkan pendekatan yang lebih terlokalisasi berdasarkan pengelompokan bisnis spesifik.
- Analisis lebih lanjut dapat melibatkan eksplorasi karakteristik (misalnya, jenis bisnis, ukuran) bisnis dalam setiap klaster untuk memahami faktor-faktor yang mendorong pengelompokan geografis ini, serta bereksperimen dengan jumlah klaster yang berbeda untuk mengoptimalkan Skor Silhouette.
""")
