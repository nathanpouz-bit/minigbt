import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

DIM = 384
index = faiss.IndexFlatL2(DIM)

DATA_FILE = "data.json"


def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def embed(text):
    return EMBED_MODEL.encode([text])[0]


def rebuild_index():
    global index
    index = faiss.IndexFlatL2(DIM)

    data = load_data()

    if len(data) == 0:
        return

    vectors = []
    for item in data:
        vectors.append(embed(item["user"]))

    vectors = np.array(vectors).astype("float32")
    index.add(vectors)


def search(query, k=1):
    data = load_data()
    if len(data) == 0:
        return None

    q_vec = np.array([embed(query)]).astype("float32")

    D, I = index.search(q_vec, k)

    if I[0][0] == -1:
        return None

    return data[I[0][0]]


def add_memory(user, bot):
    data = load_data()
    data.append({"user": user, "bot": bot})

    save_data(data)

    # update index
    rebuild_index()
