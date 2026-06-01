import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sistem Rekomendasi E-Commerce", layout="wide")

st.title("🛍️ Capstone Project Dicoding CC26-PRU466")
st.markdown("### Sistem Rekomendasi Produk E-commerce menggunakan Neural Collaborative Filtering")

# ============ GANTI DENGAN URL API SETELAH DEPLOY KE RENDER ============
API_URL = "https://ncf-recommendation-api.onrender.com"  # <-- GANTI NANTI

# Cek koneksi ke API
@st.cache_data(ttl=60)
def check_api():
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        return resp.status_code == 200
    except:
        return False

api_ready = check_api()

if not api_ready:
    st.error("🚨 API tidak tersedia. Pastikan backend sudah di-deploy.")
    st.info("URL API saat ini: " + API_URL)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🔍 Cari Rekomendasi")
    
    # Ambil daftar user dari API
    try:
        resp = requests.get(f"{API_URL}/users")
        users = resp.json().get("users", [])
        selected_user = st.selectbox("Pilih User ID", users)
    except:
        selected_user = st.text_input("User ID", "A1PSUH0U1FPQ6R")
    
    top_k = st.slider("Jumlah rekomendasi", 5, 20, 10)
    
    if st.button("🚀 Dapatkan Rekomendasi", type="primary"):
        with st.spinner("Memproses..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={"user_id": selected_user, "top_k": top_k},
                    timeout=30
                )
                if response.status_code == 200:
                    st.session_state.recommendations = response.json()
                    st.success("Berhasil!")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown')}")
            except Exception as e:
                st.error(f"Gagal: {e}")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Insight Dataset")
    st.markdown("""
    - **Total User:** ~8 unique users
    - **Total Produk:** ~8 unique products
    - **Rating Range:** 0-5 stars (denormalized)
    - **Model:** Neural Collaborative Filtering
    """)
    
    st.subheader("🏗️ Komponen Kustom")
    st.markdown("""
    - ✅ **Custom Layer:** `ProductInteractionLayer`
    - ✅ **Custom Loss:** `root_mean_squared_error_loss`
    """)

with col2:
    if "recommendations" in st.session_state:
        recs = st.session_state.recommendations
        st.subheader(f"🎯 Rekomendasi untuk User: `{recs['user_id']}`")
        
        df_recs = pd.DataFrame(recs["recommendations"])
        
        # Chart
        fig = px.bar(
            df_recs, 
            x="product_id", 
            y="predicted_rating",
            title="Predicted Rating per Produk",
            color="predicted_rating",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel
        st.dataframe(df_recs, use_container_width=True)
    else:
        st.info("👈 Pilih user dan klik 'Dapatkan Rekomendasi'")

# Footer
st.markdown("---")
st.caption("© 2026 - Capstone Project Dicoding CC26-PRU466 | NCF with Custom Layer & Custom Loss")
