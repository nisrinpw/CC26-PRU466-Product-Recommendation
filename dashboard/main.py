import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# 🔥 WAJIB: Registrasi Custom Layer (Seperti yang sudah kamu buat)
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
    model = tf.keras.models.load_model(
        "saved_model/ncf_product_recommendation.keras", # Pastikan path ini benar
        custom_objects={
            'ProductInteractionLayer': ProductInteractionLayer,
            'root_mean_squared_error_loss': root_mean_squared_error_loss
        }
    )
    scaler = joblib.load("saved_model/rating_scaler.pkl")
    df = pd.read_csv("data/cleaned_sample_data_scaled.csv")
    
    # Mapping data (sederhanakan untuk hemat memori)
    unique_users = df['user_id'].unique()
    user_to_encoded = {user: idx for idx, user in enumerate(unique_users)}
    encoded_to_product = {idx: pid for idx, pid in enumerate(df['product_id'].unique())}
    
    return model, scaler, user_to_encoded, encoded_to_product

# Panggil fungsi cache (hanya berjalan sekali)
model, scaler, user_mapping, product_mapping = load_recommendation_system()
st.success("Model siap!")
