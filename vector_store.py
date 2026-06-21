import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "data.json"

model_embed = SentenceTransformer("all-MiniLM-L6-v2")


def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)


def embed(text):
    return np.array(model_embed.encode(text))


def add_memory(user, bot):
    data = load_data()
    data.append({"user": user, "bot": bot})
    save_data(data)


def search(query):
    data = load_data()
    if len(data) == 0:
        return None

    q = embed(query)

    best_score = -1
    best_item = None

    for item in data:
        v = embed(item["user"])
        score = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))

        if score > best_score:
            best_score = score
            best_item = item

    if best_score > 0.75:
        return best_item

    return None


def rebuild_index():
    # compatibilité (ne fait rien ici)
    pass
