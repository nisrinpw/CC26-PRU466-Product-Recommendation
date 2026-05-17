from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Product Recommendation API",
    description="API untuk Sistem Rekomendasi E-Commerce menggunakan NCF",
    version="1.0"
)

# 1. REGISTRASI CUSTOM COMPONENTS (WAJIB)
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

# 2. LOAD MODEL DAN SCALER
MODEL_PATH = os.path.join("saved_model", "ncf_product_recommendation.keras")
SCALER_PATH = os.path.join("saved_model", "rating_scaler.pkl")

try:
    # Memuat model lengkap dengan custom objects
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'ProductInteractionLayer': ProductInteractionLayer,
            'root_mean_squared_error_loss': root_mean_squared_error_loss
        }
    )
    # Memuat scaler untuk inverse transform
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Model dan Scaler berhasil dimuat!")
except Exception as e:
    print(f"[ERROR] Gagal memuat model/scaler: {e}")

# 3. SCHEMA REQUEST & ENDPOINTS
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

@app.get("/")
def read_root():
    return {"message": "Welcome to NCF Product Recommendation API! Endpoint aktif: /recommend"}

@app.post("/recommend")
def get_recommendations(req: RecommendationRequest):
    try:
        # MVP Simulasi, Memprediksi skor untuk 100 produk dummy pertama 
        num_dummy_products = 100
        
        # Membuat array input
        user_input = np.array([req.user_id] * num_dummy_products)
        product_input = np.array(range(num_dummy_products))

        # Prediksi hasil
        predictions_scaled = model.predict([user_input, product_input], verbose=0)

        # Mengembalikan prediksi ke skala bintang (1.0 - 5.0) menggunakan scaler
        predictions_original = scaler.inverse_transform(predictions_scaled)

        # Mendapatkan indeks produk dengan rating prediksi tertinggi
        top_indices = predictions_original.flatten().argsort()[-req.top_k:][::-1]

        # Menyusun format output JSON
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                "product_encoded_id": int(product_input[idx]),
                "predicted_rating_star": round(float(predictions_original[idx][0]), 2)
            })

        return {
            "status": "success",
            "user_id_requested": req.user_id,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))