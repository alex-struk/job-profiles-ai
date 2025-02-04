from scripts.cleaning.main import generate_job_profiles_md
from scripts.llm.main import get_model_id
from scripts.util.main import loadCSVData
from transformers import AutoTokenizer

def analyze_profiles(profiles):
    # Analyze headings and lengths
    all_headings = set()
    lengths = []
    heading_counts = {}
    
    for profile in profiles:
        # Get length statistics
        length = len(profile)
        lengths.append(length)
        
        # Extract headings
        lines = profile.split('\n')
        for line in lines:
            # Check for '**' format headings
            if line.strip().startswith('**') and ':**' in line:
                # Extract the heading part between ** and :**
                heading = line.split(':**')[0].strip('* ').upper()
                all_headings.add(heading)
                heading_counts[heading] = heading_counts.get(heading, 0) + 1
    
    # Print analysis
    print("\n=== Dataset Analysis ===")
    print(f"\nTotal number of profiles: {len(profiles)}")
    print(f"Average profile length: {sum(lengths)/len(lengths):.2f} characters")
    print(f"Max profile length: {max(lengths)} characters")
    print(f"Min profile length: {min(lengths)} characters")
    
    print("\nUnique headings found:")
    for heading in sorted(all_headings):
        percentage = (heading_counts[heading] / len(profiles)) * 100
        print(f"- {heading}: {heading_counts[heading]} profiles ({percentage:.1f}%)")


def analyze_token_lengths(profiles, tokenizer):
    token_lengths = []
    for profile in profiles:
        tokens = tokenizer(profile, truncation=False)
        token_lengths.append(len(tokens['input_ids']))
    
    print("\n=== Token Length Analysis ===")
    print(f"Average tokens per profile: {sum(token_lengths)/len(token_lengths):.2f}")
    print(f"Max tokens: {max(token_lengths)}")
    print(f"Min tokens: {min(token_lengths)}")

def runMain():
    df = loadCSVData()
    profiles = generate_job_profiles_md(df)
    tokenizer = AutoTokenizer.from_pretrained(get_model_id())

    print('Analyzing data set...')
    analyze_profiles(profiles)

    print('Analyzing token lengths..')
    analyze_token_lengths(profiles, tokenizer)
