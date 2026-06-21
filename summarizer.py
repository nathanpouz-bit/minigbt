from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_history(history_text):
    if len(history_text) < 200:
        return history_text

    summary = summarizer(history_text, max_length=80, min_length=20, do_sample=False)
    return summary[0]["summary_text"]
