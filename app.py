import os
import time
import threading
import pandas as pd
from flask import Flask, jsonify, send_file
from flair.data import Sentence
from flair.models import SequenceTagger
from newsapi import NewsApiClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Constants
API_KEY = '7d5669c6af444e7f876a888678d11a19'
NEWS_SOURCES = ["the-times-of-india", "google-news-in", "news24", "abc-news", "cnn"]
NER_TAGS = ['GPE', 'ORG', 'PERSON']
BERT_MODEL = "bert-base-uncased"
OUTPUT_FILENAME = 'rex.csv'
FETCH_INTERVAL = 86400  # 24 hours in seconds

# Initialize the news API client
api = NewsApiClient(api_key=API_KEY)

app = Flask(__name__)

def fetch_news_data(sources):
    """Fetch news articles from specified sources."""
    dfs = []
    for source in sources:
        response = api.get_everything(sources=source)
        if response['status'] == 'ok':
            articles_df = pd.DataFrame(response['articles'])
            articles_df['source'] = articles_df['source'].apply(lambda x: x['name'])
            dfs.append(articles_df)
        else:
            print(f"Failed to fetch data from {source}: {response.get('message')}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_ner_model():
    """Load the pre-trained NER model."""
    return SequenceTagger.load("flair/ner-english-ontonotes-large")

def populate_ner_tags(row, tagger, ner_tags):
    """Extract NER tags and populate the corresponding columns."""
    if pd.isnull(row['content']):
        return row

    sentence = Sentence(row['content'])
    tagger.predict(sentence)
    entities = sentence.get_spans('ner')

    for entity in entities:
        tag_type = entity.get_label('ner').value
        if tag_type in ner_tags:
            if row[tag_type] == "":
                row[tag_type] = entity.text
            else:
                row[tag_type] += f", {entity.text}"

    return row

def process_ner_tags(df, tagger, ner_tags):
    """Process the DataFrame to extract NER tags."""
    for tag in ner_tags:
        df[tag] = ""
    return df.apply(populate_ner_tags, axis=1, tagger=tagger, ner_tags=ner_tags)

def load_classification_model(model_name):
    """Load the pre-trained classification model."""
    return AutoModelForSequenceClassification.from_pretrained(model_name)

def categorize_articles(df, model, tokenizer_name):
    """Categorize articles using the specified model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer_name, device=0 if device == "cuda" else -1)
    
    categories = []
    for text in df["description"].fillna("").astype(str):
        category = classifier(text, top_k=1)[0]["label"]
        categories.append(category)
    
    return categories

def run_script():
    """Function to run the script."""
    df = fetch_news_data(NEWS_SOURCES)
    
    if df.empty:
        print("No data fetched. Exiting.")
        return

    tagger = load_ner_model()
    df = process_ner_tags(df, tagger, NER_TAGS)

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    classification_model = load_classification_model('dima806/news-category-classifier-distilbert')
    df["Article Tag"] = categorize_articles(df, classification_model, BERT_MODEL)

    if os.path.exists(OUTPUT_FILENAME):
        previous_df = pd.read_csv(OUTPUT_FILENAME)
        df = pd.concat([previous_df, df], ignore_index=True)

    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Data saved to {OUTPUT_FILENAME}")

def periodic_task():
    """Periodic task to run the script every 24 hours."""
    while True:
        run_script()
        time.sleep(FETCH_INTERVAL)

@app.route('/run-script', methods=['GET'])
def run_script_endpoint():
    """Endpoint to run the script."""
    run_script()
    return jsonify({"message": f"Data saved to {OUTPUT_FILENAME}"}), 200

@app.route('/download-latest-excel', methods=['GET'])
def download_latest_excel():
    """Endpoint to download the latest Excel file."""
    if os.path.exists(OUTPUT_FILENAME):
        return send_file(OUTPUT_FILENAME, as_attachment=True)
    else:
        return jsonify({"message": "File not found."}), 404

if __name__ == "__main__":
    # Start the periodic task in a separate thread
    threading.Thread(target=periodic_task, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
