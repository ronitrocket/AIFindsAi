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
import json
import pickle
import faiss
import numpy as np
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
os.chdir("/Users/ronittaleti/AIFindsAi/aisearchtoolproject")

print("Current working directory:", os.getcwd())


conn = sqlite3.connect("ai_tools.db", check_same_thread=False)
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

faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)

app = Flask(__name__)

# Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

    try:
        redis_client.ft("model_index").info()  # Check if index exists
    except:
        print("sus")
        redis_client.ft("model_index").create_index([
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": VECTOR_DIMENSION,
                "DISTANCE_METRIC": "COSINE"
            })
        ], definition=IndexDefinition(prefix=["model:"], index_type=IndexType.HASH))

def store_data_in_redis_faiss(model_id, text):
    embedding = embedding_model.encode([text], normalize_embeddings=True)[0]
    faiss_index.add(np.array([embedding], dtype=np.float32))
    redis_client.set(model_id, json.dumps({"text": text}))
    # redis_client.hset(f"model:{model_id}", mapping={
    #     "model_id": model_id, 
    #     "embedding": embedding.astype(np.float32).tobytes()
    # })

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
                        store_data_in_redis_faiss(model_id, readme_text)
                os.remove(readme_path)
            except Exception as e:
                print(f"Error fetching README for {model_id}: {str(e)}")
    
    print("All models cached successfully.")

def generate_embedding(text):
    return embedding_model.encode([text], normalize_embeddings=True)[0].astype(np.float32).tobytes()

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

    print(f"FAISS index size: {faiss_index.ntotal}")



    query_vector = embedding_model.encode([query], normalize_embeddings=False)[0]
    print(f"Query vector shape: {query_vector.shape}, FAISS index shape: {faiss_index.d}")

    distances, indices = faiss_index.search(query_vector.reshape(1, -1), k=10)  # Reshape for batch search

    print(f"Distances: {distances}, Indices: {indices}")
    # distances, indices = faiss_index.search(np.array([query_vector], dtype=np.float32), k=10)
    
    results = []
    for idx in indices[0]:
        if idx < 0: continue  # Skip invalid indices
        cursor.execute("SELECT model_id FROM models LIMIT 1 OFFSET ?", (int(idx),))
        model_row = cursor.fetchone()
        if model_row:
            model_id = model_row[0]
            metadata = redis_client.get(model_id)
            if metadata:
                results.append({"model_id": model_id, "description": json.loads(metadata)})

    print(f"Distances: {distances}, Indices: {indices}")
    print(f"FAISS index size: {faiss_index.ntotal}")

    valid_indices = [idx for idx in indices[0] if idx >= 0]  # Remove invalid indices
    print(f"Valid Indices: {valid_indices}")

    if not valid_indices:
        print("FAISS returned no valid indices.")

    print(json.dumps(results, indent=4))


    return jsonify(results)

    
    # query_vector = generate_embedding(query)  # Returns a list


    # print("Query vector bytes length:", len(query_vector))

    # search_query = Query("*=>[KNN 10 @embedding $vec AS score]").sort_by("score").paging(0, 10).dialect(2)
    # results = redis_client.ft("model_index").search(search_query, query_params={"vec": query_vector})
    # print("Raw results:", results)

    # search_query = redis.commands.search.query.Query(f"*=>[KNN {10} @embedding $vec AS score]").sort_by("score").paging(0, 10).dialect(2)

    # results = redis_client.ft("model_index").search(search_query, query_params={"vec": query_vector})

    # # matches = [(doc["word"], float(doc["score"])) for doc in results.docs]
    # matches = []
    # for doc in results.docs:
    #     model_id = doc.id
    #     score = float(doc.score)
    #     matches.append((model_id, score))
    # print(f"Matches: {len(matches)}")
    # print("Top matches:")
    # for word, score in matches:
    #     print(f"{word}: {score}")

    # return [(doc["id"], float(doc["score"])) for doc in results.docs]


    # models = []
    # for doc in results.docs:
    #     models.append({"model_id": doc["model_id"], "score": float(doc["score"])})

    # models = sorted(models, key=lambda x: x["score"])

    # return jsonify({"query": query, "results": models}) if models else jsonify({"error": "No relevant models found"}), 404

    # return [(doc["word"], float(doc["score"])) for doc in results.docs]

    # try:
    #     search_result = redis_client.ft("model_index").search(
    #         search_query, query_params={"query_vector": query_vector}
    #     )

    #     # Process results
    #     models = []
    #     for doc in search_result.docs:
    #         models.append({"model_id": doc.id, "score": float(doc.score)})

    #     # Sort by similarity (low score = more relevant)
    #     models = sorted(models, key=lambda x: x["score"])

    #     return jsonify({"query": query, "results": models}) if models else jsonify({"error": "No relevant models found"}), 404

    # except Exception as e:
    #     print(f"Search error: {e}")
    #     return jsonify({"error": "Redis search failed"}), 500

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
        store_data_in_redis_faiss(model_id, readme_text)
        count += 1
        print(f"current model #: {count}", end="\r", flush=True)
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
    print(f"Loading models from SQLite into Redis.")
    # load_from_sqlite_to_redis()
    # faiss.write_index(faiss_index, "faiss_index.bin")
    print(f"FAISS index size: {faiss_index.ntotal}")
    if os.path.exists("faiss_index.bin"):
        faiss_index = faiss.read_index("faiss_index.bin")
    print(f"FAISS index size: {faiss_index.ntotal}")    
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