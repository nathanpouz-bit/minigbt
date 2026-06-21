import json
import os
import numpy as np

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
    # 🔥 embedding ultra simple (fallback)
    return np.array([hash(w) % 1000 for w in text.lower().split()])


def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def add_memory(user, bot):
    data = load_data()
    data.append({"user": user, "bot": bot})
    save_data(data)


def search(query):
    data = load_data()
    if not data:
        return None

    q = embed(query)

    best = None
    best_score = -1

    for item in data:
        v = embed(item["user"])
        score = similarity(q, v)

        if score > best_score:
            best_score = score
            best = item

    if best_score > 0.5:
        return best

    return None


def rebuild_index():
    pass
