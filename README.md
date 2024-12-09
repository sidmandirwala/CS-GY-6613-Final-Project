# RAG System with Multi-Source Data Crawling

This project implements a Retrieval-Augmented Generation (RAG) system that crawls data from multiple sources (GitHub, Medium, LinkedIn), processes it, and provides a question-answering interface through both API and Gradio UI.

## Important things to Highlight
- We have used a GPT-2 small model and finetuned it on 50 Epochs because we couldnt run the larger model locally after finetuning it. The output might not be that accurate because of gpt2 being a smaller model.
- We used Kaggle to finetune it and have attached the code for it aswell (Finetuning.ipynb). For you too retrain it, you may require the wandb key, the hugging face key is already there in the ipynb.
- The finetuned model has been pushed to https://huggingface.co/prathamssaraf/finetuned-gpt2_50_new-peft
  
## Prerequisites

- Docker
- Python 3.x
- MongoDB
- Qdrant Vector Database

## Installation

### 1. Set up Databases

First, start MongoDB and Qdrant using Docker
This part of the code can be directly done using docker-compose.yml file or using the following commands:

```bash
# Start MongoDB
docker pull mongo
docker run -d --name mongodb -p 27017:27017 mongo

# Start Qdrant
docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant
```
### The data collection pipeline can be directly run using start.sh script or the following commands can be executed.
### 2. Data Collection

Run the following scrapers in separate terminals to collect data from different sources:

```bash
python crawler/scraper_github.py
python crawler/scraper_medium.py
python crawler/scraper_linkedin.py
```

### 3. Data Processing

Process the collected data through the cleaning and feature pipelines:

```bash
python cleaning.py
python feautre_pipeline/feature_pipeline.py
python feautre_pipeline/feature_extension_pipeline.py
```

### 4. Load Question-Answer Pairs

As this implementation doesn't use the OpenAI API, manually push the Question-Answer pairs to Qdrant:

```bash
python push_qna_to_qdrant.py
```

## Running the Application

Start the following components in separate terminals:

1. Start the model server:
```bash
python serve_model.py
```

2. Start the RAG API:
```bash
python rag_pipeline/rag_api.py
```

3. Start the Gradio interface:
```bash
python rag_pipeline/rag_gradio_app.py
```

The Gradio interface will be available at: http://0.0.0.0:7860

## Project Structure

```
.
├── crawler/
│   ├── scraper_github.py
│   ├── scraper_medium.py
│   └── scraper_linkedin.py
├── feature_pipeline/
│   ├── feature_pipeline.py
│   └── feature_extension_pipeline.py
├── rag_pipeline/
│   ├── rag_api.py
│   └── rag_gradio_app.py
├── cleaning.py
├── push_qna_to_qdrant.py
└── serve_model.py
```

## Notes

- Make sure all required Python dependencies are installed
- Each component should be run in a separate terminal
- Ensure all databases are running before starting the application
- Monitor the Docker containers for any potential issues
