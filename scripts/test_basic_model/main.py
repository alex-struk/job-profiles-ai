from scripts.cleaning.main import generate_job_profiles_md
from scripts.llm.main import get_model_id, getModel
from scripts.util.main import loadCSVData
import torch
from transformers import AutoTokenizer

# Modified evaluation function for your data structure
def evaluate_model_performance(model, tokenizer, profiles):
    print("\n=== Starting Model Evaluation ===")
    model.eval()
    
    results = {
        'perplexities': [],
        'section_coverage': [],
        'sections_found_details': []
    }
    
    # Only evaluate first 5 profiles for efficiency
    for i, profile in enumerate(profiles[:5], 1):
        print(f"\nEvaluating Profile {i}/5 {'='*40}")
        print(f"Profile length: {len(profile)} characters")
        
        try:
            # 1. Test Perplexity on full profile
            inputs = tokenizer(
                profile, 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(model.device)
            
            print(f"Tokens in profile: {len(inputs['input_ids'][0])}")
            
            # Calculate perplexity
            inputs['labels'] = inputs['input_ids'].clone()
            with torch.no_grad():
                outputs = model(**inputs)
                perplexity = torch.exp(outputs.loss).item()
                results['perplexities'].append(perplexity)
                print(f"Perplexity score: {perplexity:.2f}")
            
            # 2. Test Generation with just the title
            # Extract title from profile
            title = profile.split('\n')[0]  # Get first line
            title = title.replace('# **Title:** ', '').strip()  # Extract just the title text
            prompt = f"Profile: \n\n # **Title:** {title}"

            print("\nGenerating new profile from prompt...")
            print(f"Prompt: {prompt}")

            # Tokenize prompt
            prompt_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(model.device)

            # Generate new text from prompt
            generated = model.generate(
                input_ids=prompt_inputs['input_ids'],
                attention_mask=prompt_inputs['attention_mask'],
                max_new_tokens=2000,  # Increased to allow for full profile
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id 
            )
            generated_text = tokenizer.decode(generated[0])
            
            # Remove the prompt from the generated text
            prompt_length = len(prompt)
            generated_text = generated_text[prompt_length+3:]

            print("\nGenerated text sample (first 300 chars):")
            print(generated_text[:300] + "...\n")
            
            # Check section coverage in generated text
            common_sections = [
                "CLOSE DATE", "DIVISION", "JOB CATEGORY", "JOB TYPE", 
                "JOB URL", "LOCATION", "MINISTRY/ORGANIZATION", 
                "POSITION CLASSIFICATION", "SALARY RANGE", "QUALIFICATIONS",
                "UNION", "WORK OPTIONS", "APPLICATION PROCESS", "ORGANIZATION"
            ]
            
            found_sections = []
            for section in common_sections:
                if section in generated_text.upper():
                    found_sections.append(section)
            
            coverage = len(found_sections) / len(common_sections)
            results['section_coverage'].append(coverage)
            results['sections_found_details'].append(found_sections)
            
            print("Sections Analysis:")
            print(f"Found {len(found_sections)} out of {len(common_sections)} sections")
            print("Found sections:", ", ".join(found_sections))
            print(f"Coverage score: {coverage:.2%}")
            
        except Exception as e:
            print(f"Error processing profile {i}: {str(e)}")
            continue
    
    # Final results
    if results['perplexities']:
        avg_perplexity = sum(results['perplexities']) / len(results['perplexities'])
        avg_coverage = sum(results['section_coverage']) / len(results['section_coverage'])
        
        print("\n=== Final Evaluation Results ===")
        print(f"Average Perplexity: {avg_perplexity:.2f}")
        print(f"Average Section Coverage: {avg_coverage:.2%}")
        print("\nPerplexity scores per profile:")
        for i, score in enumerate(results['perplexities'], 1):
            print(f"Profile {i}: {score:.2f}")
        
        print("\nSection coverage per profile:")
        for i, (coverage, sections) in enumerate(zip(results['section_coverage'], 
                                                   results['sections_found_details']), 1):
            print(f"\nProfile {i}:")
            print(f"Coverage: {coverage:.2%}")
            print(f"Found sections: {', '.join(sections)}")
    
    return results


def runMain():
    df = loadCSVData()
    profiles = generate_job_profiles_md(df)

    # Load base model for evaluation
    print("Loading base model for evaluation...")
    model = getModel()

    # Run evaluation on base model
    print("Evaluating base model performance...")
    tokenizer = AutoTokenizer.from_pretrained(get_model_id())
    evaluate_model_performance(model, tokenizer, profiles)

if __name__ == "__main__":
    runMain()