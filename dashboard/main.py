import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Sistem Rekomendasi E-Commerce", layout="wide")

st.title("🛍️ Capstone Project Dicoding CC26-PRU466")
st.markdown("### Sistem Rekomendasi Produk E-commerce menggunakan Neural Collaborative Filtering")

# ============ REGISTER CUSTOM COMPONENTS ============
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

# ============ LOAD MODEL & DATA (CACHED) ============
@st.cache_resource
def load_model_and_data():
    # Path ke file (relative dari folder dashboard)
    MODEL_PATH = "../saved_model/ncf_product_recommendation.keras"
    SCALER_PATH = "../saved_model/rating_scaler.pkl"
    DATA_PATH = "../data/cleaned_sample_data_scaled.csv"
    
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
    
    # Mapping user_id string ke integer
    unique_users = df['user_id'].unique()
    unique_products = df['product_id'].unique()
    
    user_to_encoded = {user: idx for idx, user in enumerate(unique_users)}
    encoded_to_user = {idx: user for user, idx in user_to_encoded.items()}
    product_to_encoded = {product: idx for idx, product in enumerate(unique_products)}
    encoded_to_product = {idx: product for product, idx in product_to_encoded.items()}
    
    return model, scaler, df, user_to_encoded, encoded_to_user, product_to_encoded, encoded_to_product, len(unique_products)

# Load semua data (dijalankan sekali saja)
with st.spinner("Memuat model dan data..."):
    model, scaler, df, user_to_encoded, encoded_to_user, product_to_encoded, encoded_to_product, num_products = load_model_and_data()

st.success("✅ Model dan data berhasil dimuat!")

# ============ SIDEBAR ============
with st.sidebar:
    st.header("🔍 Parameter Rekomendasi")
    
    # Pilih user
    users_list = list(user_to_encoded.keys())
    selected_user = st.selectbox("Pilih User ID", users_list)
    
    top_k = st.slider("Jumlah Rekomendasi", 5, 20, 10)
    
    if st.button("🚀 Dapatkan Rekomendasi", type="primary"):
        with st.spinner("Memproses rekomendasi..."):
            user_encoded = user_to_encoded[selected_user]
            
            # Prediksi untuk semua produk
            user_input = np.array([user_encoded] * num_products)
            product_input = np.array(list(range(num_products)))
            
            predictions_scaled = model.predict([user_input, product_input], verbose=0)
            predictions_original = scaler.inverse_transform(predictions_scaled)
            
            # Ambil top_k produk
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
            st.success(f"✅ Berhasil! Mendapatkan {top_k} rekomendasi")

# ============ MAIN CONTENT ============
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Informasi Dataset")
    st.metric("Total Transaksi", len(df))
    st.metric("Unique Users", df['user_id'].nunique())
    st.metric("Unique Products", df['product_id'].nunique())
    
    with st.expander("Preview Data"):
        st.dataframe(df.head(5), use_container_width=True)
    
    st.subheader("🏗️ Komponen Kustom")
    st.markdown("""
    - ✅ **Custom Layer:** `ProductInteractionLayer`
    - ✅ **Custom Loss:** `root_mean_squared_error_loss`
    """)

with col2:
    if 'recommendations' in st.session_state and st.session_state['recommendations']:
        recs = st.session_state['recommendations']
        st.subheader(f"🎯 Top {len(recs)} Rekomendasi untuk User: `{st.session_state['selected_user']}`")
        
        df_recs = pd.DataFrame(recs)
        
        # Chart
        fig = px.bar(
            df_recs,
            x="product_id",
            y="predicted_rating",
            title="Predicted Rating per Produk",
            color="predicted_rating",
            color_continuous_scale="Viridis",
            text="predicted_rating"
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, xaxis_title="Product ID", yaxis_title="Predicted Rating (1-5)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel
        st.dataframe(df_recs, use_container_width=True)
        
        # Download button
        csv = df_recs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Rekomendasi (CSV)",
            data=csv,
            file_name=f"recommendations_user_{st.session_state['selected_user']}.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Pilih User ID di sidebar, lalu klik 'Dapatkan Rekomendasi'")
        
        # Contoh tampilan
        st.subheader("📈 Contoh Visualisasi")
        st.caption("Silakan pilih user dan dapatkan rekomendasi untuk melihat hasil")

# ============ FOOTER ============
st.markdown("---")
st.caption("© 2026 - Capstone Project Dicoding CC26-PRU466 | Neural Collaborative Filtering dengan Custom Layer & Custom Loss")
