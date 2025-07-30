# AssignmentSolution for LLM For Junction.

# README: Document Search and Summarization Using Large Language Models (LLM)

## Overview

This project implements a **document search and summarization system** using a combination of traditional information retrieval and advanced Large Language Models (LLMs). It allows users to search a large text corpus and receive concise summaries of relevant documents.

## Features

- **Efficient document search** with support for traditional (TF-IDF/BM25) and embedding-based approaches
- **LLM-powered summarization** (e.g., using OpenAI GPT-4 or Huggingface Transformers)
- **User-driven summary length** selection: short, medium, or long
- **Evaluation pipeline** for both search quality and summary quality (ROUGE, human assessments)
- (Optional) **Web interface** for query input, results browsing, and summary customization

## 1. Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- API access for LLMs (e.g., OpenAI API key if using GPT)
- CUDA-enabled GPU (optional but strongly recommended for embedding models)

## 2. Installation

Clone the repository and install dependencies:

```sh
git clone 
cd 
pip install -r requirements.txt
```

## 3. Project Structure

```
.
├── data_prep.py        # Dataset loading and pre-processing
├── search.py           # Document search pipeline (TF-IDF, embeddings, ranking)
├── summarize.py        # Document summarization scripts
├── evaluate.py         # Evaluation: search and summarization quality
├── app.py              # (Optional) Web interface backend (e.g., Streamlit)
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── data/               # Data storage directory (corpus, processed data, embeddings, etc.)
```

## 4. Usage

### **Step 1: Data Preparation**

Prepare and process the dataset:

```sh
python data_prep.py --input data/raw_corpus.txt --output data/cleaned_data.json
```
- Supports datasets like 20 Newsgroups, Wikipedia subsets, etc.
- Cleans and stores documents in JSON with IDs and text fields.

### **Step 2: Search (IR & Embeddings)**

To run a query and retrieve top search results:

```sh
python search.py --query "climate change effects" --top_n 5
```
- Supports traditional TF-IDF and/or embedding-based retrieval.
- Outputs top-N relevant documents for the query.

### **Step 3: Summarization**

To summarize the top search results:

```sh
python summarize.py --doc_ids 1 42 37 --length medium
```
- Use `--length` flag to select summary size (short/medium/long).
- Summaries are generated via LLM API or local model.

### **Step 4: Evaluation**

Run evaluation scripts (automated + manual):

```sh
python evaluate.py --mode search    # Evaluates search accuracy (e.g., Precision@5)
python evaluate.py --mode summary   # Evaluates ROUGE scores for summaries
```
- Also supports generating reports or exporting results.

### **Step 5: (Optional) Run Web Interface**

To launch the interactive web interface:

```sh
streamlit run app.py
```
- Search, browse, and summarize via your browser.
- Adjust summary length, navigate paginated results.

## 5. Configuration and Customization

- Edit `.env` or config files to set API keys or tweak search/summarization weights.
- To change models (e.g., use another transformer), modify imports/parameters in the corresponding scripts.
- For large datasets, use FAISS or Annoy indexing for faster embedding search.

## 6. Troubleshooting

- **API errors:** Check that your API key for OpenAI or Huggingface is loaded in your environment.
- **Performance:** For faster processing on large data, set up a GPU and use batch methods.
- **Memory issues:** Chunk large texts, reduce batch size, or process documents in smaller sets.

## 7. Credits & References

- LLMs: [OpenAI GPT-4], [Huggingface Transformers]
- Dataset: 20 Newsgroups (or your selected corpus)
- Libraries: scikit-learn, sentence-transformers, FAISS, NLTK, pandas, numpy, Streamlit
