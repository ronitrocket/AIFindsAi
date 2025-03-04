# import requests
# import redis
# from huggingface_hub import HfApi
# api = HfApi()
# models = api.list_models()
# models = list(models)
# print(len(models))
# print(models[0].modelId)

# # Redis Cache
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# # # PostgreSQL Connection
# # conn = psycopg2.connect("dbname=ai_tools user=youruser password=yourpass")
# # cursor = conn.cursor()

# # Fetch models from Hugging Face API
# def fetch_models():
#     url = "https://huggingface.co/api/models"
#     response = requests.get(url)
#     models = response.json()
    
#     for model in models[:50]:  # Limit for testing
#         model_name = model['modelId']
#         description = model.get('pipeline_tag', 'Unknown')
        
#         # Store in Redis (short-term cache)
#         redis_client.setex(model_name, 86400, str(description))  # Expires in 24 hours
        
#     #     # Store in PostgreSQL (long-term storage)
#     #     cursor.execute(
#     #         "INSERT INTO ai_models (name, description) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
#     #         (model_name, description)
#     #     )

#     # conn.commit()
#     print("Data cached successfully.")

# fetch_models()


# while True:
#     userInput = input("Say EXIT to exit, else will return all models cached: ")
#     if userInput == exit:
#         break
#     print("sus")
#     for key in redis_client.scan_iter():
#         # delete the key
#         print(redis_client.get(key))


# import requests
# import redis
# from huggingface_hub import HfApi

# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# api = HfApi()


# def fetch_and_store_models():
#     print("Fetching models from Hugging Face API...")
#     models = api.list_models()
    
#     total_models = len(models)
#     print(f"Total models found: {total_models}")
    
#     for model in models:
#         model_id = model.modelId
#         description = model.get('pipeline_tag', 'Unknown')
#         redis_client.setex(model_id, 86400, description)
    
#     print("All models cached successfully.")

# fetch_and_store_models()

# def query_models():
#     while True:
#         user_input = input("Enter model name to search (or type EXIT to quit): ")
#         if user_input.lower() == "exit":
#             break
#         model_description = redis_client.get(user_input)
#         if model_description:
#             print(f"Model: {user_input}, Description: {model_description}")
#         else:
#             print("Model not found in cache.")

# query_models()

import requests
import redis
import os
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json

VECTOR_DIMENSION = 384  # Embedding size for MiniLM

app = Flask(__name__)

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Initialize Hugging Face API
api = HfApi()

def create_redis_index():
    try:
        redis_client.execute_command(f"FT.CREATE model_index ON HASH PREFIX 1 model: "
                                     f"SCHEMA vector VECTOR HNSW 6 TYPE FLOAT32 DIM {VECTOR_DIMENSION} DISTANCE_METRIC COSINE")
        print("Redis index created.")
    except redis.exceptions.ResponseError:
        print("Redis index already exists.")

def get_embedding(text):
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return np.array(vector, dtype=np.float32).tobytes()

def store_model_embedding(model_id, description):
    vector = get_embedding(description)
    redis_client.hset(f"model:{model_id}", mapping={"vector": vector, "description": description})

# Function to fetch and store models
def fetch_and_store_models():
    print("Fetching models from Hugging Face API...")
    models = list(api.list_models(
            filter="automatic-speech-recognition"
        )
    )[-1000:]
    
    total_models = len(models)
    print(f"Total models found: {total_models}")

    corpus = []

    for model in models:
        model_id = model.modelId
        cached_readme = redis_client.get(model_id)
        if not cached_readme: 
            try:
                # model_info = api.model_info(model_id)
                readme_path = hf_hub_download(model_id, "README.md")
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_text = f.read()
                    print(readme_text)
                    if readme_text:
                        corpus.append(readme_text)
                os.remove(readme_path)
            except Exception as e:
                print(f"Error fetching README for {model_id}: {str(e)}")
        

    print(f"Total README files retrieved: {len(corpus)}")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    terms = vectorizer.get_feature_names_out()

    i = 0
    for model in models:
        model_id = model.modelId
        cached_readme = redis_client.get(model_id)

        if not cached_readme: 
            scores = tfidf_matrix[i].toarray().flatten()
            unique_parts = [terms[j] for j in scores.argsort()[-10:]]
            redis_client.setex(model_id, 86400, json.dumps(unique_parts))  # Cache for 24 hours
        i = i + 1

    # i = 0
    # for model in models:
    #     model_id = model.modelId
    #     cached_readme = redis_client.get(model_id)
    #     if not cached_readme: 
    #         try:
    #             # model_info = api.model_info(model_id)
    #             readme_path = hf_hub_download(model_id, "README.md")
    #             with open(readme_path, "r", encoding="utf-8") as f:
    #                 readme_text = f.read()
    #                 if readme_text:
    #                     store_model_embedding(model_id, readme_text);
    #             os.remove(readme_path)
    #         except Exception as e:
    #             print(f"Error fetching README for {model_id}: {str(e)}")
    #             return "README not found or error fetching it."
    #     i = i + 1

    
    print("All models cached successfully.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_model():
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    
    model_description = redis_client.get(model_name)
    if model_description:
        return jsonify({"model": model_name, "description": model_description})
    else:
        return jsonify({"error": "Model not found in cache"}), 404

if __name__ == '__main__':
    fetch_and_store_models()
    create_redis_index()
    print("\nFirst 100 Cached Entries:")
    for i, key in enumerate(redis_client.scan_iter(match="*", count=100)):
        if i >= 100:
            break
        print(f"{i+1}. {key}: {redis_client.get(key)[:200]}...")  # Print first 200 chars of each entry
    app.run(debug=True)