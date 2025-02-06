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

Run `poetry env info --path` to get interpreter path

Ctrl+Shift+P, Python->Select Interpreter, set the path.

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


**Output Example**

Prompt: 

`prompt = f"Profile: \n\n # **Title:** AI Data Science Engineer \n"`

Output:

```
 # **Title:** AI Data Science Engineer
 # **Location:** Vancouver, bc CA (Primary)
 # **Job Type:** Regular Full Time
 # **Salary Range:** $82,873.76 - $103,585.78 annually
 # **Close Date:** 2/24/2025
 # **Ministry/Organization:** BC Public Service -> Children & Family Development
 # **Division:** Ministry of Children and Family Development
 # **Position Classification:** Scientist 1
 # **Job Category:** Scientific and Professional, Scientific and Professional Leads
 # **Union:** GEU
 # **Job URL:** [Link](https://bcpublicservice.hua.hrsmart.com/hr/ats/Posting/view/119241)

**Employement Type:** Permanent

**Temporary/Coverage Opportunity:** No

**Indigenous Applicants:** Consideration for this position is available to Indigenous (First Nations, Status or Non-Status, Métis, or Inuit) applicants who self-identify as Indigenous in their application and who meet the minimum qualifications. Applicants who do not self-identify as Indigenous will also be considered.

**Organization:** This position is located at the Ministry of Children and Family Development in Vancouver.

**Application Process:** Please review the required documents section at the bottom of the posting for information on how to apply.

**The BC Public Service is committed to creating a diverse workplace to represent the population we serve and to better meet the needs of our citizens. Consider joining our team and being part of an innovative, inclusive and rewarding workplace.
We are committed to ensuring that reasonable accommodations are available throughout the hiring process, including the assessment and selection stages. Please email the individual or contact listed on the posting if you require an accommodation to fully participate in the hiring process.
The Indigenous Applicant Advisory Service is available to Canadian Indigenous (First Nations [status or non-status], Métis, or Inuit) applicants. Indigenous applicants can contact this service for personalized guidance on the BC Public Service hiring process including job applications and interviews.
The BC Public Service is an award-winning employer and offers employees competitive benefits, amazing learning opportunities and a chance to engage in rewarding work with exciting career development opportunities. For more information, please see What We Offer.
The BC Public Service is committed to creating a diverse workplace to represent the population we serve and to better meet the needs of our citizens. Consider joining our team and being part of an innovative, inclusive and rewarding workplace.
The BC Public Service is an equal opportunity employer committed to creating a diverse workforce to represent the population we serve. Applicants with disabilities may be granted accommodation at any stage of the hiring process. Please contact the hiring team at the email below if you require such an accommodation.
```

**To do**
- Setup Tensorboard/Weights & Biases to track train/grad_norm