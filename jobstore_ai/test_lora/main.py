from jobstore_ai.llm.main import get_model_id, getModel
from peft import  PeftModel
from transformers import AutoTokenizer

def generate_profile(prompt, model_path="./lora_job_adapter"):
    # Load model with adapter
    model = getModel()
    model = PeftModel.from_pretrained(model, model_path)
    tokenizer = AutoTokenizer.from_pretrained(get_model_id(),
        # model_max_length=4096  # Next power of 2 above max tokens
    )
    tokenizer.pad_token = tokenizer.eos_token
    # model.resize_token_embeddings(len(tokenizer))
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs['attention_mask'],
        max_new_tokens=3700,
        temperature=0.6,
        # discourage repeated text
        # repetition_penalty=2.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id 
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def runMain():
    # TRY LORA - Generate new profile with lora
    title = "AI Data Science Engineer"
    prompt = f"Profile: \n\n # **Title:** {title} \n"
    # **Location:** Vancouver, BC V6B 0N8 CA (Primary) \n\n **Job Type:** Regular Full Time \n\n **Salary Range:** $76,071.18 - $86,658.48 annually \n\n **Close Date:** 2/18/2025
    new_profile = generate_profile(prompt)
    print(new_profile)


if __name__ == "__main__":
    runMain()