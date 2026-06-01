from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
import os
import pandas as pd

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Product Recommendation API",
    description="API untuk Sistem Rekomendasi E-Commerce menggunakan NCF",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development, bisa ganti dengan URL Streamlit nanti
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
MODEL_PATH = "saved_model/ncf_product_recommendation.keras"
SCALER_PATH = "saved_model/rating_scaler.pkl"
DATA_PATH = "data/cleaned_sample_data_scaled.csv"

model = None
scaler = None
user_to_encoded = {}
encoded_to_user = {}
product_to_encoded = {}
encoded_to_product = {}
num_products = 0

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
    df = pd.read_csv(DATA_PATH)
    
    unique_users = df['user_id'].unique()
    unique_products = df['product_id'].unique()
    
    user_to_encoded = {user: idx for idx, user in enumerate(unique_users)}
    encoded_to_user = {idx: user for user, idx in user_to_encoded.items()}
    product_to_encoded = {product: idx for idx, product in enumerate(unique_products)}
    encoded_to_product = {idx: product for product, idx in product_to_encoded.items()}
    
    num_products = len(unique_products)
    print(f"[INFO] Loaded: {len(unique_users)} users, {num_products} products")
except Exception as e:
    print(f"[ERROR] {e}")
    
# 3. SCHEMA REQUEST & ENDPOINTS
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

@app.get("/")
def root():
    return {"status": "ready", "users": len(user_to_encoded), "products": num_products}
    
@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if model is None:
        raise HTTPException(500, "Model not loaded")
    if req.user_id not in user_to_encoded:
        raise HTTPException(404, f"User {req.user_id} not found")
    
    user_encoded = user_to_encoded[req.user_id]
    user_input = np.array([user_encoded] * num_products)
    product_input = np.array(list(range(num_products)))
    
    predictions_scaled = model.predict([user_input, product_input], verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    top_indices = predictions.flatten().argsort()[-req.top_k:][::-1]
    
    recommendations = [
        {
            "product_id": encoded_to_product[product_input[idx]],
            "predicted_rating": round(float(predictions[idx][0]), 2)
        }
        for idx in top_indices
    ]
    
    return {"user_id": req.user_id, "recommendations": recommendations}

@app.get("/users")
def get_users():
    return {"users": list(user_to_encoded.keys())}
