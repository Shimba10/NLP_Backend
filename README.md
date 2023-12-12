
## Main Steps and Motivation for Data Processing:

- Loaded data from a CSV file into a DataFrame.
- Placeholder data used for simplicity.

## Main Functionalities of the API:

1. **Summarization Endpoint (/summarize):**
   - Implements a placeholder logic for document summarization.
   - Used bart tokenizer to encode the document text.
   - facebook/bart-large-cnn model genrated the tokenizer output.
   - Decoded the model output.

2. **Text Classification Endpoint (/classify):**
   - Used nlptown/bert-base-multilingual-uncased-sentiment model for the classification the document text.
   - It will return label and score in response.

3. **Identify topics Endpoint (/identify_topics):**
   - Used dslim/bert-base-NER model and tokenzied the document text.
   - It will return reapeated text from the document.

## Key Challenges Faced:

- Searching for the correct model name.
- To write the correct logic for all 3 endpoints.

## Improvements:

- Integrate HuggingFace models for better Identify topics and summarization.
- Implement error handling for robustness.
