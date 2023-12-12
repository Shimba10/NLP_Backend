
from flask import Flask, jsonify, current_app
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

summarize_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
summarize_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
key_topic_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
key_topic_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)


document_df = pd.read_csv("text_segments.csv")
document_text = document_df["text"].values[0]

@app.route('/summarize', methods=['POST'])
def summarize():
    inputs = summarize_tokenizer.encode("summarize: " + document_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarize_model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarize_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    current_app.logger.info(f"summary :  {summary}")

    return jsonify({"summary": summary})

@app.route('/identify_topics', methods=['POST'])
def identify_topics():
    key_topics_pipeline = pipeline('ner', model=key_topic_model, tokenizer=key_topic_tokenizer)
    key_topics = key_topics_pipeline(document_text)
    key_topics_serializable = [{'word': str(topic['word']), 'score': float(topic['score'])} for topic in key_topics]
    current_app.logger.info(f"key_topics :  {key_topics_serializable}")


    return jsonify({"key_topics": key_topics_serializable})

@app.route('/classify', methods=['POST'])
def classify():

    result = classifier(document_text)
    current_app.logger.info(f"result : {result}")
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
