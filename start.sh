#!/bin/bash

# Install requirements
pip install -r requirements.txt

# Run the python3 scripts
python3 crawler/scraper_github.py
python3 crawler/scraper_medium.py
python3 crawler/scraper_linkedin.py
python3 cleaning.py
python3 feature_pipeline/feature_pipeline.py
python3 feature_pipeline/feature_pipeline_extension.py
python3 push_qna_to_qdrant.py