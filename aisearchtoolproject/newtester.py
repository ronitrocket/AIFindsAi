import redis
import random
import string
import numpy as np
from sentence_transformers import SentenceTransformer

def random_word(length=5):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def setup_redis():
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    index_name = "word_embeddings"
    
    # Check if the index exists
    try:
        r.ft(index_name).info()
    except:
        # Create vector index (HNSW)
        r.ft(index_name).create_index([
            redis.commands.search.field.VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": 384,  # Embedding size for 'all-MiniLM-L6-v2'
                "DISTANCE_METRIC": "COSINE"
            })
        ], definition=redis.commands.search.indexDefinition.IndexDefinition(prefix=["word:"], index_type=redis.commands.search.indexDefinition.IndexType.HASH))
    
    return r, index_name

def store_words(r, model, num_words=100):
    words = [random_word() for _ in range(num_words)]
    print("Generated words:", words)  # Print all generated words
    embeddings = model.encode(words, normalize_embeddings=True)  # Normalize for cosine similarity
    
    for i, word in enumerate(words):
        r.hset(f"word:{i}", mapping={
            "word": word,
            "embedding": embeddings[i].astype(np.float32).tobytes()
        })

def search_words(r, model, index_name, query_word, top_k=10):
    query_embedding = model.encode([query_word], normalize_embeddings=True)[0].astype(np.float32).tobytes()
    query = redis.commands.search.query.Query(f"*=>[KNN {top_k} @embedding $vec AS score]").sort_by("score").paging(0, top_k).dialect(2)
    results = r.ft(index_name).search(query, query_params={"vec": query_embedding})
    return [(doc["word"], float(doc["score"])) for doc in results.docs]

def main():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    r, index_name = setup_redis()
    store_words(r, model)
    
    user_word = input("Enter a word to search: ")
    matches = search_words(r, model, index_name, user_word)
    
    print("Top matches:")
    for word, score in matches:
        print(f"{word}: {score}")

if __name__ == "__main__":
    main()
