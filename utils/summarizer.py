from transformers import pipeline

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    # Hugging Face models have input length limits (1024 tokens), so we truncate long input
    max_input_length = 1024
    text = text[:max_input_length]

    summary_list = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary_list[0]["summary_text"]
