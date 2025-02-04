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


** Output Example **

Prompt: `"Profile: \n\n # **Title:** Specialist, Creative \n\n **Location:** Vancouver, BC V6B 0N8 CA (Primary) \n\n **Job Type:** Regular Full Time \n\n **Salary Range:** $76,071.18 - $86,658.48 annually \n\n **Close Date:** 2/18/2025`

Output:

```
Profile:

 # **Title:** Specialist, Creative

 **Location:** Vancouver, BC V6B 0N8 CA (Primary)

 **Job Type:** Regular Full Time

 **Salary Range:** $76,071.18 - $86,658.48 annually

 **Close Date:** 2/18/2025

 **Job ID:** 10000357

 **Organization:** BC Public Service

 **Ministry:** Ministry of Citizens' Services

 **Job Description:**

**Job Summary**

The BC Public Service is looking for a creative, innovative and collaborative individual to join the Digital Services team in the Ministry of Citizens’ Services.

The Digital Services team is responsible for the design, development and delivery of digital services for the people of British Columbia. We are looking for a creative, innovative and collaborative individual to join our team.

The Digital Services team is responsible for the design, development and delivery of digital services for the people of British Columbia. We are looking for a creative, innovative and collaborative individual to join our team.

**The Opportunity**

The Specialist, Creative will be responsible for the design and development of digital services for the people of British Columbia. You will work with a team of designers, developers, and product managers to create and deliver digital services that meet the needs of our users.

**The Role**

As a Specialist, Creative, you will be responsible for:

* Designing and developing digital services that meet the needs of our users.
* Working with a team of designers, developers, and product managers to create and deliver digital services.
* Collaborating with stakeholders to understand their needs and requirements.
* Ensuring that digital services are accessible, usable, and secure.
* Staying up-to-date with the latest trends and technologies in the field of digital services.

**The Qualifications**

To be successful in this role, you will need to have:

* A degree or diploma in design, computer science, or a related field.
* Experience in designing and developing digital services.
* Strong communication and collaboration skills.
* The ability to work in a fast-paced and dynamic environment.
* A passion for creating digital services that make a difference in the lives of our users.

**The Benefits**

In addition to a competitive salary, the BC Public Service offers a comprehensive benefits package that includes:

* Extended health and dental coverage.
* A pension plan.
* A flexible work environment.
* Professional development opportunities.

**The Application Process**

To apply for this opportunity, please submit your resume and cover letter through the BC Public Service website.
```
