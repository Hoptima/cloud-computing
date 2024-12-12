from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from PropertyRAG import PropertyRAG
import sys
import json
import os
from google.cloud import storage
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
# from flask_cors import CORS

app = Flask(__name__)

# CORS(app)

# Firebase initialization
dbkey = credentials.Certificate("dbkey.json")
firebase_admin.initialize_app(dbkey)
db = firestore.client()

# Google Cloud Storage setup
credentials = service_account.Credentials.from_service_account_file('key.json')
client = storage.Client(credentials=credentials, project='qwerty88')
bucket_name = 'cpstnml'  # Nama bucket Anda
model_directory = 'model/'
bucket = client.bucket(bucket_name)

# Temporary directory setup
base_directory = os.getcwd()
temp_directory = os.path.join(base_directory, 'temp')

if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

# Files to download
files_to_download = ['databaru.csv', 'numeric_model.h5', 'text_model.h5', 'tfidf_vectorizer.pkl']

for file_name in files_to_download:
    blob = bucket.blob(model_directory + file_name)
    temp_file_path = os.path.join(temp_directory, file_name)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    blob.download_to_filename(temp_file_path)

# Load models and data
def load_keras_models():
    text_model = load_model('temp/text_model.h5')
    numeric_model = load_model('temp/numeric_model.h5')
    return text_model, numeric_model

def load_tfidf_vectorizer():
    with open('temp/tfidf_vectorizer.pkl', 'rb') as f:
        return pickle.load(f)

def load_property_data():
    return pd.read_csv('temp/databaru.csv', encoding='utf-8')

text_model, numeric_model = load_keras_models()
tfidf_vectorizer = load_tfidf_vectorizer()
df = load_property_data()

rag = PropertyRAG(df, text_model, numeric_model, tfidf_vectorizer)

# Helper functions
def display_regex_results(query, rag):
    kamar = rag.extract_numeric_requirements(query)[0] or 0
    wc = rag.extract_numeric_requirements(query)[1] or 0
    parkir = rag.extract_numeric_requirements(query)[2] or 0
    max_price = rag.extract_numeric_requirements(query)[3] or float('inf')
    luas_tanah = rag.extract_numeric_requirements(query)[4] or 0
    luas_bangunan = rag.extract_numeric_requirements(query)[5] or 0

    print(f"Jumlah Kamar: {kamar if kamar > 0 else 'Tidak Terdeteksi'}")
    print(f"Jumlah Kamar Mandi: {wc if wc > 0 else 'Tidak Terdeteksi'}")
    print(f"Luas Tanah: {luas_tanah} m²" if luas_tanah > 0 else "Tidak Terdeteksi")
    print(f"Slot Parkir: {parkir if parkir > 0 else 'Tidak Terdeteksi'}")
    if max_price != float('inf'):
        print(f"Harga Maksimal: Rp {max_price:,.0f}")
    else:
        print("Harga Maksimal: Tidak Terdeteksi")
    print(f"Luas Bangunan: {luas_bangunan} m²" if luas_bangunan > 0 else "Tidak Terdeteksi")
    location = rag.find_location(query)
    print(f"Lokasi yang Terdeteksi: {location if location else 'Tidak Terdeteksi'}")

def get_recommendations(query, rag):
    recommendations = rag.get_recommendations(query)
    if not recommendations.empty:
        print("Rekomendasi Properti:")
        print(recommendations)
    else:
        print("Maaf, tidak ada rekomendasi yang sesuai dengan kriteria Anda.")
    return recommendations

def chat_recommendations(query, rag):
    print("Chat Rekomendasi Properti")
    print("Input: ", query)
    user_message = {"role": "user", "content": query}
    print(f"User: {user_message['content']}")
    assistant_message = {"role": "assistant", "content": "Berikut adalah rekomendasi properti untuk Anda:"}
    print(f"Assistant: {assistant_message['content']}")
    recommendations = get_recommendations(query, rag)
    display_regex_results(query, rag)

def get_next_prediction_id():
    collection_ref = db.collection('property_recommendations')
    docs = collection_ref.stream()
    return sum(1 for _ in docs) + 1

def generate_unique_id():
    return str(uuid.uuid4())

def save_to_firestore(output):
    unique_id = generate_unique_id()
    document_name = f"{unique_id}"
    document_ref = db.collection('property_recommendations').document(document_name)
    output["prediction_id"] = unique_id
    document_ref.set(output)
    print(f"Data berhasil disimpan ke Firestore dengan nama dokumen: {document_name} dan ID unik: {unique_id}")

# Flask endpoint
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        query = request.json.get('query', '')
        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Get recommendations
        recommendations = rag.get_recommendations(query)
        recommendations_json = recommendations.to_dict(orient="records")

        # Extract regex results
        kamar = rag.extract_numeric_requirements(query)[0] or 0
        wc = rag.extract_numeric_requirements(query)[1] or 0
        parkir = rag.extract_numeric_requirements(query)[2] or 0
        max_price = rag.extract_numeric_requirements(query)[3] or float('inf')
        luas_tanah = rag.extract_numeric_requirements(query)[4] or 0
        luas_bangunan = rag.extract_numeric_requirements(query)[5] or 0
        location = rag.find_location(query)
        regex_results = {
            "Jumlah Kamar": kamar if kamar > 0 else "Tidak Terdeteksi",
            "Jumlah Kamar Mandi": wc if wc > 0 else "Tidak Terdeteksi",
            "Slot Parkir": parkir if parkir > 0 else "Tidak Terdeteksi",
            "Harga Maksimal": f"Rp {max_price:,.0f}" if max_price != float('inf') else "Tidak Terdeteksi",
            "Luas Tanah": f"{luas_tanah} m²" if luas_tanah > 0 else "Tidak Terdeteksi",
            "Luas Bangunan": f"{luas_bangunan} m²" if luas_bangunan > 0 else "Tidak Terdeteksi",
            "Lokasi": location if location else "Tidak Terdeteksi",
        }

        output = {
            "assistant_message": "Berikut adalah rekomendasi properti untuk Anda:",
            "recommendations": recommendations_json,
            "regex_results": regex_results,
        }

        # Prepare Firestore data with top 10 recommendations
        firestore_output = {
            "assistant_message": "Berikut adalah rekomendasi properti untuk Anda:",
            "recommendations": recommendations_json[:10],  # Limit to top 10
            "regex_results": regex_results,
        }

        # Save top 10 data to Firestore
        save_to_firestore(firestore_output)

        # Return full recommendations as response
        return jsonify(output)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

