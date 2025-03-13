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
from sentence_transformers import SentenceTransformer
import sqlite3
import pickle
import numpy as np
os.chdir("/Users/ronittaleti/AIFindsAi/aisearchtoolproject")

print("Current working directory:", os.getcwd())


conn = sqlite3.connect("ai_tools.db")
cursor = conn.cursor()
# cursor.execute("DROP TABLE IF EXISTS models")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS models (
        model_id TEXT PRIMARY KEY,
        readme TEXT,
        embedding BLOB
    )
""")
conn.commit()

VECTOR_DIMENSION = 384

app = Flask(__name__)

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=False)

# Initialize Hugging Face API
api = HfApi()

def create_redis_index():
    try:
        # Attempt to drop the existing index (DD drops the associated documents too)
        redis_client.execute_command("FT.DROPINDEX", "model_index", "DD")
        print("Existing index dropped.")
    except redis.exceptions.ResponseError as e:
        print("Index did not exist or could not be dropped:", e)

    # Create the index with the desired schema
    try:
        redis_client.execute_command(
            "FT.CREATE", "model_index", "ON", "HASH", "PREFIX", "1", "model:",
            "SCHEMA", "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32", "DIM", str(VECTOR_DIMENSION),
            "DISTANCE_METRIC", "COSINE"
        )
        print("Redis index created.")
    except redis.exceptions.ResponseError as e:
        print("Error creating index:", e)

def store_embedding_in_redis(model_id, embedding):
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
    redis_client.hset(f"model:{model_id}", mapping={"embedding": embedding_bytes})

# Function to fetch and store models
def fetch_and_store_models():
    print("Fetching models from Hugging Face API...")
    models = list(api.list_models(
            filter="automatic-speech-recognition"
        )
    )
    
    total_models = len(models)
    
    i = 0
    for model in models:
        i = i+1
        print(f"Total models found: {total_models} | Current model: {i}")
        model_id = model.modelId
        cached_readme = redis_client.hgetall(model_id)
        cursor.execute("SELECT 1 FROM models WHERE model_id = ?", (model_id,))
        exists = cursor.fetchone() is not None
        if not cached_readme or not exists: 
            try:
                model_info = api.model_info(model_id)
                readme_path = hf_hub_download(model_id, "README.md")
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_text = f.read()
                    if readme_text:
                        embedding = generate_embedding(readme_text)
                        store_in_sqlite(model_id, readme_text)
                        store_embedding_in_redis(model_id, embedding)
                os.remove(readme_path)
            except Exception as e:
                print(f"Error fetching README for {model_id}: {str(e)}")
    
    print("All models cached successfully.")

# Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    return embedding_model.encode(text).tolist()  # Convert to list for storage

def store_in_sqlite(model_id, readme_text):
    embedding = generate_embedding(readme_text)
    embedding_blob = pickle.dumps(embedding)  # Serialize embedding
    
    cursor.execute("""
        INSERT INTO models (model_id, readme, embedding)
        VALUES (?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET readme = ?, embedding = ?
    """, (model_id, readme_text, embedding_blob, readme_text, embedding_blob))
    
    conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search_model():
    query = request.args.get('model_name')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Generate embedding for user query
    query_embedding = generate_embedding(query)  # Returns a list

    print("Query embedding:", query_embedding)

    # Convert to Redis-friendly format (FLOAT32)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    print("Query vector bytes length:", len(query_vector))

    # search_result = redis_client.execute_command(
    #     "FT.SEARCH", "model_index",
    #     "*=>[KNN 5 @embedding $query_vector AS vector_score]",  # Correct KNN syntax
    #     "SORTBY", "vector_score", "ASC",
    #     "PARAMS", "2", "query_vector", query_vector,
    #     "DIALECT", "2"  # Ensure correct RediSearch 2.x dialect
    # )

    search_result = redis_client.execute_command(
        "FT.SEARCH", "model_index",
        "*=>[KNN 5 @embedding $query_vector AS vector_score]",  
        "SORTBY", "vector_score", "ASC",
        "PARAMS", "2", "query_vector", query_vector,
        "RETURN", "1", "vector_score",  # Explicitly return the score field
        "DIALECT", "2"
    )
    # Process results
    models = []
    for i in range(2, len(search_result), 2):  # Skip Redis metadata
        model_id = search_result[i]
        score = float(search_result[i+1][1])  # Extract similarity score
        models.append({"model_id": model_id, "score": score})

    # Sort by similarity (low score = more relevant)
    models = sorted(models, key=lambda x: x["score"])

    print("Query vector:", query_vector)
    print("Search result:", search_result)

    if models:
        return jsonify({"query": query, "results": models})
    else:
        return jsonify({"error": "No relevant models found"}), 404

def load_from_sqlite_to_redis():
    """
    Load all model data from SQLite into Redis.
    This can be used to repopulate Redis without refetching from Hugging Face.
    """
    cursor.execute("SELECT model_id, readme, embedding FROM models")
    rows = cursor.fetchall()
    count = 0
    for row in rows:
        model_id, readme_text, embedding_blob = row
        key = f"model:{model_id}"
        # Store both readme and embedding in Redis
        redis_client.hset(key, mapping={"readme": readme_text.encode("utf-8"), "embedding": embedding_blob})
        count += 1
    print(f"Loaded {count} models from SQLite into Redis.")

# def search_model():
#     model_name = request.args.get('model_name')
#     if not model_name:
#         return jsonify({"error": "Model name required"}), 400
    
#     model_description = redis_client.hgetall(model_name)
#     if model_description:
#         return jsonify({"model": model_name, "description": model_description})
#     else:
#         return jsonify({"error": "Model not found in cache"}), 404

if __name__ == '__main__':
    #fetch_and_store_models()
    load_from_sqlite_to_redis()
    create_redis_index()
    # print(list(redis_client.scan_iter("model:*")))
    # sample_key = next(redis_client.scan_iter("model:*"), None)
    # if sample_key:
    #     print(redis_client.hget(sample_key, "embedding"))
    sample_text = "Test input"
    embedding = generate_embedding(sample_text)
    print(f"Embedding shape: {len(embedding)}")
    print(redis_client.execute_command("FT.INFO", "model_index"))
    # print(type(query_vector), len(query_vector))
    # for key in redis_client.scan_iter("model:*"):
    #     data = redis_client.hget(key, 'embedding')
        
    #     print(pickle.loads(data) if isinstance(data, bytes) else data)
    print(redis_client.execute_command("FT._LIST"))
    # print("\nFirst 100 Cached Entries:")
    # for i, key in enumerate(redis_client.scan_iter(match="*", count=100)):
    #     if i >= 100:
    #         break
    #     print(f"{i+1}. {key}: {redis_client.hgetall(key)[:200]}...")  # Print first 200 chars of each entry
    app.run(debug=True)