import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Sistem Rekomendasi E-Commerce", layout="wide")

st.title("🛍️ Capstone Project Dicoding CC26-PRU466")
st.markdown("### Sistem Rekomendasi Produk E-commerce menggunakan Neural Collaborative Filtering")

# 🔥 WAJIB: Registrasi Custom Layer
@tf.keras.utils.register_keras_serializable()
class ProductInteractionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ProductInteractionLayer, self).__init__(**kwargs)
    def call(self, inputs):
        user_vector, item_vector = inputs
        return user_vector * item_vector

@tf.keras.utils.register_keras_serializable()
def root_mean_squared_error_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# 🔥 SOLUSI UTAMA: Gunakan cache untuk memuat model hanya SEKALI
@st.cache_resource
def load_recommendation_system():
    # Path relatif di server
    MODEL_PATH = "saved_model/ncf_product_recommendation.keras"
    SCALER_PATH = "saved_model/rating_scaler.pkl"
    DATA_PATH = "data/cleaned_sample_data_scaled.csv"
    
    # Cek apakah file ada (untuk debugging)
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan di: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"❌ Scaler tidak ditemukan di: {SCALER_PATH}")
        st.stop()
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Data tidak ditemukan di: {DATA_PATH}")
        st.stop()
    
    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'ProductInteractionLayer': ProductInteractionLayer,
            'root_mean_squared_error_loss': root_mean_squared_error_loss
        }
    )
    
    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Mapping data (lengkap)
    unique_users = df['user_id'].unique()
    unique_products = df['product_id'].unique()
    
    user_to_encoded = {user: idx for idx, user in enumerate(unique_users)}
    encoded_to_user = {idx: user for user, idx in user_to_encoded.items()}
    
    product_to_encoded = {product: idx for idx, product in enumerate(unique_products)}
    encoded_to_product = {idx: product for product, idx in product_to_encoded.items()}
    
    return model, scaler, user_to_encoded, encoded_to_user, product_to_encoded, encoded_to_product, df

# Panggil fungsi cache (hanya berjalan sekali)
with st.spinner("Memuat model dan data..."):
    try:
        model, scaler, user_to_encoded, encoded_to_user, product_to_encoded, encoded_to_product, df = load_recommendation_system()
        st.success("✅ Model dan data berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat: {str(e)}")
        st.stop()

# ============ SIDEBAR ============
with st.sidebar:
    st.header("🔍 Parameter Rekomendasi")
    
    users_list = list(user_to_encoded.keys())
    selected_user = st.selectbox("Pilih User ID", users_list)
    top_k = st.slider("Jumlah Rekomendasi", 5, 20, 10)
    
    if st.button("🚀 Dapatkan Rekomendasi", type="primary"):
        with st.spinner("Memproses rekomendasi..."):
            user_encoded = user_to_encoded[selected_user]
            num_products = len(product_to_encoded)
            
            # Prediksi untuk semua produk
            user_input = np.array([user_encoded] * num_products)
            product_input = np.array(list(range(num_products)))
            
            predictions_scaled = model.predict([user_input, product_input], verbose=0)
            predictions_original = scaler.inverse_transform(predictions_scaled)
            
            # Ambil top_k
            top_indices = predictions_original.flatten().argsort()[-top_k:][::-1]
            
            recommendations = []
            for idx in top_indices:
                product_encoded = product_input[idx]
                original_product_id = encoded_to_product[product_encoded]
                recommendations.append({
                    "product_id": original_product_id,
                    "predicted_rating": round(float(predictions_original[idx][0]), 2)
                })
            
            st.session_state['recommendations'] = recommendations
            st.session_state['selected_user'] = selected_user
            st.success(f"✅ Berhasil! {top_k} rekomendasi didapatkan")

# ============ MAIN CONTENT ============
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Informasi Dataset")
    st.metric("Total Transaksi", len(df))
    st.metric("Unique Users", df['user_id'].nunique())
    st.metric("Unique Products", df['product_id'].nunique())
    
    with st.expander("Preview Data"):
        st.dataframe(df.head(5))

with col2:
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        st.subheader(f"🎯 Top {len(recs)} Rekomendasi untuk User: `{st.session_state['selected_user']}`")
        
        df_recs = pd.DataFrame(recs)
        
        # Chart
        fig = px.bar(df_recs, x="product_id", y="predicted_rating", 
                     title="Predicted Rating per Produk", color="predicted_rating",
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel
        st.dataframe(df_recs)
    else:
        st.info("👈 Pilih User ID dan klik 'Dapatkan Rekomendasi'")

st.markdown("---")
st.caption("© 2026 - Capstone Project Dicoding CC26-PRU466 | NCF with Custom Layer & Custom Loss")
