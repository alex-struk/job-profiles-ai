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

**Techinical Information**

```
=== Dataset Analysis ===
Total number of profiles: 122
Average profile length: 6436.07 characters
Max profile length: 17230 characters
Min profile length: 554 characters

=== Token Length Analysis ===
Average tokens per profile: 1420.12
Max tokens: 3668
Min tokens: 175

Unique headings found:
- ACCOUNTABILITIES: 4 profiles (3.3%)
- APPLICATION PROCESS: 117 profiles (95.9%)
- CLOSE DATE: 122 profiles (100.0%)
- COMPETENCIES: 2 profiles (1.6%)
- DIVISION: 122 profiles (100.0%)
- EDUCATION REQUIREMENTS: 30 profiles (24.6%)
- JOB CATEGORY: 122 profiles (100.0%)
- JOB TYPE: 122 profiles (100.0%)
- JOB URL: 122 profiles (100.0%)
- LOCATION: 122 profiles (100.0%)
- MINISTRY/ORGANIZATION: 122 profiles (100.0%)
- ORGANIZATION: 117 profiles (95.9%)
- POSITION CLASSIFICATION: 122 profiles (100.0%)
- PROVISOS: 11 profiles (9.0%)
- QUALIFICATIONS: 85 profiles (69.7%)
- SALARY RANGE: 122 profiles (100.0%)
- SKILLS: 1 profiles (0.8%)
- TEMPORARY END DATE: 16 profiles (13.1%)
- UNION: 95 profiles (77.9%)
- WORK OPTIONS: 96 profiles (78.7%)


Train Log:

{'loss': 1.5074, 'grad_norm': 0.582065224647522, 'learning_rate': 3e-05, 'epoch': 0.02}
{'loss': 1.4034, 'grad_norm': 0.41912639141082764, 'learning_rate': 3e-05, 'epoch': 0.05}                                                                                                                        
{'loss': 1.553, 'grad_norm': 0.4319896697998047, 'learning_rate': 3e-05, 'epoch': 0.07}                                                                                                                          
{'loss': 1.5143, 'grad_norm': 0.48047006130218506, 'learning_rate': 3e-05, 'epoch': 0.1}                                                                                                                         
{'loss': 1.4974, 'grad_norm': 0.4509294927120209, 'learning_rate': 3e-05, 'epoch': 0.12}                                                                                                                         
{'loss': 1.3976, 'grad_norm': 0.3554573357105255, 'learning_rate': 3e-05, 'epoch': 0.15}                                                                                                                         
{'loss': 1.5802, 'grad_norm': 0.46818286180496216, 'learning_rate': 3e-05, 'epoch': 0.17}                                                                                                                        
{'loss': 1.3091, 'grad_norm': 0.3856395483016968, 'learning_rate': 3e-05, 'epoch': 0.2}                                                                                                                          
{'loss': 1.2953, 'grad_norm': 0.45510080456733704, 'learning_rate': 3e-05, 'epoch': 0.22}                                                                                                                        
{'loss': 1.3841, 'grad_norm': 0.44376158714294434, 'learning_rate': 3e-05, 'epoch': 0.24}                                                                                                                        
{'loss': 1.3034, 'grad_norm': 0.5284692049026489, 'learning_rate': 3e-05, 'epoch': 0.27}                                                                                                                         
{'loss': 1.2603, 'grad_norm': 0.4819232225418091, 'learning_rate': 3e-05, 'epoch': 0.29}                                                                                                                         
{'loss': 1.3129, 'grad_norm': 0.44072121381759644, 'learning_rate': 3e-05, 'epoch': 0.32}                                                                                                                        
{'loss': 1.3776, 'grad_norm': 0.44111040234565735, 'learning_rate': 3e-05, 'epoch': 0.34}                                                                                                                        
{'loss': 1.1802, 'grad_norm': 0.4438681900501251, 'learning_rate': 3e-05, 'epoch': 0.37}                                                                                                                         
{'loss': 1.2812, 'grad_norm': 0.5231294631958008, 'learning_rate': 3e-05, 'epoch': 0.39}                                                                                                                         
{'loss': 1.202, 'grad_norm': 0.41496542096138, 'learning_rate': 3e-05, 'epoch': 0.41}                                                                                                                            
{'loss': 1.132, 'grad_norm': 0.5294812321662903, 'learning_rate': 3e-05, 'epoch': 0.44}                                                                                                                          
{'loss': 1.0352, 'grad_norm': 0.5079116821289062, 'learning_rate': 3e-05, 'epoch': 0.46}                                                                                                                         
{'loss': 1.1925, 'grad_norm': 0.6866111159324646, 'learning_rate': 3e-05, 'epoch': 0.49}                                                                                                                         
{'loss': 1.2936, 'grad_norm': 0.5611897110939026, 'learning_rate': 3e-05, 'epoch': 0.51}                                                                                                                         
{'loss': 0.984, 'grad_norm': 0.5108393430709839, 'learning_rate': 3e-05, 'epoch': 0.54}                                                                                                                          
{'loss': 1.1989, 'grad_norm': 0.4576336443424225, 'learning_rate': 3e-05, 'epoch': 0.56}                                                                                                                         
{'loss': 1.0708, 'grad_norm': 0.554589569568634, 'learning_rate': 3e-05, 'epoch': 0.59}                                                                                                                          
{'loss': 0.973, 'grad_norm': 0.3788383901119232, 'learning_rate': 3e-05, 'epoch': 0.61}                                                                                                                          
{'loss': 1.2652, 'grad_norm': 0.5780847668647766, 'learning_rate': 3e-05, 'epoch': 0.63}                                                                                                                         
{'loss': 1.0483, 'grad_norm': 0.6358178853988647, 'learning_rate': 3e-05, 'epoch': 0.66}                                                                                                                         
{'loss': 0.939, 'grad_norm': 0.4848395586013794, 'learning_rate': 3e-05, 'epoch': 0.68}                                                                                                                          
{'loss': 0.9714, 'grad_norm': 0.5357986688613892, 'learning_rate': 3e-05, 'epoch': 0.71}                                                                                                                         
{'loss': 1.0265, 'grad_norm': 0.6332830190658569, 'learning_rate': 3e-05, 'epoch': 0.73}                                                                                                                         
{'loss': 1.0337, 'grad_norm': 0.5683214664459229, 'learning_rate': 3e-05, 'epoch': 0.76}                                                                                                                         
{'loss': 1.0422, 'grad_norm': 0.71308833360672, 'learning_rate': 3e-05, 'epoch': 0.78}                                                                                                                           
{'loss': 0.974, 'grad_norm': 0.5032374262809753, 'learning_rate': 3e-05, 'epoch': 0.8}                                                                                                                           
{'loss': 1.0645, 'grad_norm': 0.505253791809082, 'learning_rate': 3e-05, 'epoch': 0.83}                                                                                                                          
{'loss': 0.9442, 'grad_norm': 0.44400230050086975, 'learning_rate': 3e-05, 'epoch': 0.85}                                                                                                                        
{'loss': 1.073, 'grad_norm': 0.5997743010520935, 'learning_rate': 3e-05, 'epoch': 0.88}                                                                                                                          
{'loss': 0.939, 'grad_norm': 0.4592284858226776, 'learning_rate': 3e-05, 'epoch': 0.9}                                                                                                                           
{'loss': 0.933, 'grad_norm': 0.49040645360946655, 'learning_rate': 3e-05, 'epoch': 0.93}                                                                                                                         
{'loss': 0.9357, 'grad_norm': 0.49374568462371826, 'learning_rate': 3e-05, 'epoch': 0.95}                                                                                                                        
{'loss': 0.9494, 'grad_norm': 0.5021252036094666, 'learning_rate': 3e-05, 'epoch': 0.98}                                                                                                                         
{'loss': 1.0106, 'grad_norm': 0.540763795375824, 'learning_rate': 3e-05, 'epoch': 1.0}                                                                                                                           
{'loss': 0.8568, 'grad_norm': 0.46591728925704956, 'learning_rate': 3e-05, 'epoch': 1.02}                                                                                                                        
{'loss': 0.9183, 'grad_norm': 0.46174710988998413, 'learning_rate': 3e-05, 'epoch': 1.05}                                                                                                                        
{'loss': 0.8586, 'grad_norm': 0.5998430848121643, 'learning_rate': 3e-05, 'epoch': 1.07}                                                                                                                         
{'loss': 1.0191, 'grad_norm': 0.4691358506679535, 'learning_rate': 3e-05, 'epoch': 1.1}                                                                                                                          
{'loss': 1.0052, 'grad_norm': 0.5307166576385498, 'learning_rate': 3e-05, 'epoch': 1.12}                                                                                                                         
{'loss': 1.0485, 'grad_norm': 0.47975966334342957, 'learning_rate': 3e-05, 'epoch': 1.15}                                                                                                                        
{'loss': 0.8345, 'grad_norm': 0.5486552715301514, 'learning_rate': 3e-05, 'epoch': 1.17}                                                                                                                         
{'loss': 0.9452, 'grad_norm': 0.5215041637420654, 'learning_rate': 3e-05, 'epoch': 1.2}                                                                                                                          
{'loss': 0.8791, 'grad_norm': 0.5353925824165344, 'learning_rate': 3e-05, 'epoch': 1.22}                                                                                                                         
{'loss': 0.7527, 'grad_norm': 0.5157656073570251, 'learning_rate': 3e-05, 'epoch': 1.24}                                                                                                                         
{'loss': 0.9648, 'grad_norm': 0.4893474578857422, 'learning_rate': 3e-05, 'epoch': 1.27}                                                                                                                         
{'loss': 0.8882, 'grad_norm': 0.5190998911857605, 'learning_rate': 3e-05, 'epoch': 1.29}                                                                                                                         
{'loss': 0.8094, 'grad_norm': 0.47773754596710205, 'learning_rate': 3e-05, 'epoch': 1.32}                                                                                                                        
{'loss': 0.9111, 'grad_norm': 0.6673206090927124, 'learning_rate': 3e-05, 'epoch': 1.34}                                                                                                                         
{'loss': 0.897, 'grad_norm': 0.6530534625053406, 'learning_rate': 3e-05, 'epoch': 1.37}                                                                                                                          
{'loss': 1.009, 'grad_norm': 0.6562174558639526, 'learning_rate': 3e-05, 'epoch': 1.39}                                                                                                                          
{'loss': 0.8017, 'grad_norm': 0.4964613914489746, 'learning_rate': 3e-05, 'epoch': 1.41}                                                                                                                         
{'loss': 0.9633, 'grad_norm': 0.48765698075294495, 'learning_rate': 3e-05, 'epoch': 1.44}                                                                                                                        
{'loss': 0.9865, 'grad_norm': 0.5542242527008057, 'learning_rate': 3e-05, 'epoch': 1.46}
```

**To do**
- Setup Tensorboard/Weights & Biases to track train/grad_norm
