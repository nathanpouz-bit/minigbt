from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import os

MODEL_NAME = "microsoft/DialoGPT-medium"
DATA_FILE = "logs/conversations.json"
OUTPUT_DIR = "model_v2"


def load_data():
    if not os.path.exists(DATA_FILE):
        return []

    with open(DATA_FILE, "r") as f:
        return json.load(f)


def train():
    data = load_data()

    if len(data) < 20:
        print("Pas assez de données pour entraîner")
        return

    dataset = Dataset.from_list([
        {"text": f"User: {d['user']}\nBot: {d['bot']}"}
        for d in data
    ])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    dataset = dataset.map(tokenize)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ modèle mis à jour")
