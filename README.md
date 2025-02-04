# Mistral-7B LoRA Training Pipeline for Job Profiles

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Poetry](https://img.shields.io/badge/packaging-poetry-cyan)

End-to-end system for scraping Canadian public sector job postings, generating structured profiles, and fine-tuning Mistral-7B with LoRA adapters.

## Features
- **BC Government Job Scraper**: Targeted crawler for bcpublicservice.hrsmart.com
- **Profile Generation**: CSV-to-Markdown conversion with schema enforcement
- **4-bit QLoRA Training**: Optimized for single GPU (12GB+ VRAM)
- **Domain-Specific Evaluation**: Perplexity scoring + section coverage analysis
- **Dynamic Dataset Handling**: Automatic sequence length optimization

## Installation
```
poetry install
poetry shell
```

## Usage

### 1. Data Collection
```
python scripts/scraping/main.py
python scripts/cleaning/main.py
```

**Output Structure**:
`bc_jobs.csv → cleaned_data.csv → job_profiles.md`

### 2. Basic Model Evaluation
```
python scripts/test_basic_model/main.py
```

### 3. Dataset Evaluation
```
python scripts/analzye_data_set/main.py
```

### 4. Training

```
python scripts/llm/main.py
```


### 5. Profile Generation with LoRa

```
python scripts/test_lora/main.py
```



