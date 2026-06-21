import json
import os

FILE = "data.json"

def load_data():
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(FILE, "w") as f:
        json.dump(data, f)

def add_example(user, bot):
    data = load_data()
    data.append({"user": user, "bot": bot})
    save_data(data)


# 🧠 NOUVEAU : recherche simple dans la mémoire
def find_best_match(user_input):
    data = load_data()

    best_score = 0
    best_answer = None

    for item in data:
        if user_input.lower() in item["user"].lower():
            return item["bot"]

    return None
