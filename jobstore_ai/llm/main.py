from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback


from jobstore_ai.cleaning.main import generate_job_profiles_md
from jobstore_ai.util.main import loadCSVData
from huggingface_hub import login

import torch
from torch.utils.data import Dataset

isAvailable=torch.cuda.is_available()

print('Torch version: '+torch.__version__)

if(not isAvailable):
    raise BaseException("Cuda not available")

def get_key(key_name):
    with open('.env') as f:
        for line in f:
            if '=' in line:
                name, value = line.strip().split('=', 1)
                if name == key_name:
                    return value
    return None

login(token=get_key("HUGGINGFACE_HUB_API_KEY"))
# Base model configuration
model_id = "mistralai/Mistral-7B-v0.1"

def get_model_id():
    global model_id
    return model_id

QUICK_TEST = False

# Quantization config for 12GB VRAM
quantization_config = BitsAndBytesConfig(
    # Enables 4-bit quantization, which significantly reduces memory usage
    # The weights remain in this compressed 4-bit state while sitting in GPU memory.
    # When the model needs to perform calculations, the 4-bit weights are temporarily decompressed back to 16-bit format using a scaling factor
    load_in_4bit=True, 
    # Sets the computational data type to 16-bit floating point. 
    # This determines what precision is used during the actual computations, while keeping the weights in 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16, 
    # Specifies NormalFloat4 quantization scheme, which is optimized for normal distributions found in model weights. 
    # This typically provides better performance than regular FP4 quantization
    bnb_4bit_quant_type='nf4',
    # Enables nested quantization, which performs a second quantization pass on the already quantized weights to save an additional 0.4 bits per parameter
    # can save memory but may reduce performance slightly.
    bnb_4bit_use_double_quant=True
)

# Training arguments optimized for 12GB
training_args = TrainingArguments(
    output_dir="./lora_job_adapter",
    # num_train_epochs=1 if QUICK_TEST else 2, # use max_steps instead
    per_device_train_batch_size=2 if QUICK_TEST else 6,
    # Increase if using small batch sizes
    gradient_accumulation_steps=4 if QUICK_TEST else 1,
    # 2e-5 to 5e-5 is typically effective
    learning_rate=1e-4 if QUICK_TEST else 3e-5,
    # BF16 is a 16-bit floating-point format specifically designed for machine learning
    # BF16 is recommended over FP16 as it provides better training stability and doesn't require gradient scaling
    fp16=False,
    bf16=True,
    logging_steps=5 if QUICK_TEST else 1,
    # Gradient clipping, helps prevent gradient explosion during training by scaling down gradients when they exceed the specified threshold
    # Balances gradient stability with learning efficiency
    # Larger models (e.g., 7B+ parameters) often require stricter clipping (lower values like 0.3) to prevent instability
    # Smaller LoRA adapters (rank ≤32) may tolerate higher thresholds (0.5-1.0)
    # Sudden spikes in grad_norm warrant lower max_grad_norm
    # Lower learning rates (e.g., 2e-5 to 4e-4) pair well with higher clipping values (0.5-1.0), while aggressive rates need stricter clipping
    # Monitor the grad_norm metric - if it consistently stays below 50% of your threshold, consider increasing it slightly to allow faster convergence
    # If clipping occurs >20% of steps: lower threshold or learning rate
    max_grad_norm=1.0,
    # 10% of max_steps
    warmup_ratio=0.1,
    # smaller datasets typically need fewer steps. The formula for calculating max_steps is:
    # max_steps = (num_samples / effective_batch_size[batch_size*gradient_accumulation]) * num_epochs
    # Increasing max_steps beyond what's needed for complete data passes means the model will see samples multiple times
    max_steps = 20 if QUICK_TEST else int(120/(6)*3),
    # learning rate schedule type - how learning rate changes through the training process
    lr_scheduler_type="constant",
    # Updates model weights based on loss gradients
    # adamw - Adaptive Moment Estimation with Weight Decay
    # use paged_ variant if running out of memory
    optim="paged_adamw_8bit",
    # When to run evaluation - 'epoch', 'steps', or 'no'
    # Evaluates model performance on a separate validation dataset
    eval_strategy="no",
    load_best_model_at_end=False,
    # control model checkpoint saving during training
    save_strategy="no" if QUICK_TEST else "steps",
    save_steps=15, # once per epoch
    gradient_checkpointing=True
)

# # Define LoRA configuration
lora_config = LoraConfig(
    # Rank - controls the complexity of LoRA adaptations. Higher values (16-64) enable better learning but increase memory usage, 
    # while lower values (4-8) are more efficient but may limit capacity
    r=4 if QUICK_TEST else 64,
    # Scaling factor that determines adaptation strength. Typically set equal to or double the rank value. Higher values increase learning impact
    # Some models have it as 1/2 of r (like mistral - investigate this)
    lora_alpha=8 if QUICK_TEST else 32,
    # Specifies which layers to adapt
    target_modules=["q_proj", "v_proj"] if QUICK_TEST else [
        "q_proj",
        "v_proj",
        "k_proj", 
        "o_proj",   # Important for output generation, final transformation in the self-attention mechanism
        # ----
        # "lm_head",  # Critical for vocabulary generation - vocabulary space adaptations are rarely needed
        # "gate_proj",
        # "up_proj",
        # "down_proj"
        ],
    # Controls regularization (0.0-0.1). Higher values help prevent overfitting, especially with smaller datasets
    # Dropout regularization is a technique used in neural networks to prevent overfitting by randomly deactivating 
    # neurons during training. This forces the network to learn robust, generalizable features rather 
    # than memorizing noise in the training data.
    # Higher values = more generalization
    lora_dropout=0.0 if QUICK_TEST else 0.08,
    # Controls bias parameter training. Use "none" to skip bias training, "all" for all biases, or "lora_only" for only LoRA layer biases
    # A bias is an additional learnable parameter in neural networks that helps adjust the output by adding a constant value to the weighted input. 
    # Think of it like shifting the activation function left or right to better fit the data
    bias="none",
    # Model architecture type
    task_type="CAUSAL_LM",
    # Set to True only when using the model for inference to optimize performance
    # inference_mode= True,
)

model = None
def getModel():
    global model
    if not model:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            # Automatically manages model distribution across available GPUs or CPU. 
            # This is particularly useful for large models that might not fit on a single GPU
            device_map="auto",
            # Sets the default data type for non-quantized parts of the model to 16-bit floating point.
            torch_dtype=torch.float16
        )
    return model

def prepare_dataset(profiles):
    formatted_data = []
    for profile in profiles:
        title = profile.split('\n')[0].replace('# **Title:** ', '').strip()
        prompt = f"Profile: \n\n # **Title:** {title}"
        
        formatted_data.append({
            "prompt": prompt,
            "completion": profile
        })
    return formatted_data

def setup_model_for_training():
    # load base model
    model = getModel()

    # Gradient checkpointing is a memory optimization technique that trades computation time for reduced memory usage during training
    # if QUICK_TEST:
        
    # Saves GPU memory during training by not storing intermediate states
    # Required when using gradient checkpointing as the two features are incompatible
    # While caching can speed up inference by reusing previous computations, it's not needed during training and can consume unnecessary memory,
    model.gradient_checkpointing_enable() # gradient_checkpointing_kwargs={"use_reentrant": False}
    model.config.use_cache = False
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()

    # Prepares model for quantized training by converting critical layers (LayerNorm, embeddings) 
    # to float32 and configures gradient checkpointing for memory-efficient training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


class JobProfileDataset(Dataset):
    def __init__(self, profiles, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []
        
        working_profiles = profiles[:10] if QUICK_TEST else profiles
        augmented_profiles = augment_profiles(working_profiles)
        working_profiles = working_profiles + augmented_profiles
        
        # --- PROMPT HANDLING ---
        # Tokenize the fixed prompt template separately
        prompt = "Profile: \n\n"
        prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        self.prompt_ids = prompt_enc['input_ids'][0]
        self.prompt_length = len(self.prompt_ids)

        # --- DYNAMIC LENGTH CALCULATION ---
        # Determine maximum sequence length needed (prompt + longest profile + EOS)
        max_content_length = 0
        tokenized_profiles = []
        
        for profile in working_profiles:
            profile_enc = tokenizer(
                profile,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=False
            )
            profile_ids = profile_enc['input_ids'][0]
            tokenized_profiles.append(profile_ids)
            
            # Calculate total length for this profile (prompt + content + EOS)
            total_length = self.prompt_length + len(profile_ids) + 1
            if total_length > max_content_length:
                max_content_length = total_length

        # Cap at model's maximum context length (4096 for Mistral)
        self.max_length = min(max_content_length, 4096)

        # --- DATA PACKAGING ---
        # Process tokenized profiles with calculated max_length
        for profile_ids in tokenized_profiles:
            # Construct full sequence: prompt + profile + EOS
            full_content = torch.cat([
                self.prompt_ids,
                profile_ids,
                torch.tensor([self.tokenizer.eos_token_id])
            ])
            
            # Handle overflow with smart truncation:
            # Always preserve prompt + EOS, truncate middle content
            if len(full_content) > self.max_length:
                print('TRUNCATING')
                full_content = torch.cat([
                    full_content[:self.max_length-1],
                    torch.tensor([self.tokenizer.eos_token_id])
                ])
            
            # --- PADDING STRATEGY ---
            # Left-pad sequences
            pad_length = self.max_length - len(full_content)
            input_ids = torch.cat([
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long),
                full_content
            ])
            
            # --- MASKING LOGIC ---
            # Attention mask: 0 for padding, 1 for real content (includes prompt template + content)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            # set final eos token to 1, as it should be included
            attention_mask[-1]=1
            # Labels: -100 masks tokens from loss calculation
            labels = input_ids.clone()
            labels[:pad_length] = -100
            # The model should only learn to predict the profile text, not reproduce the prompt template itself.
            # - it only learns to generate appropriate continuations given the prompt
            # todo: this may be incorrect, since model doesn't learn the prompt/content boundary
            labels[pad_length:pad_length+self.prompt_length] = -100
            
            assert (labels[attention_mask == 0] == -100).all(), "Padding not masked"
            assert (labels[pad_length:pad_length+self.prompt_length] == -100).all(), "Prompt not masked"
            assert labels[-1] == tokenizer.eos_token_id, "EOS missing"

            # Final example packaging
            self.examples.append({
                'input_ids': input_ids, #  tokenized versions of the input text
                'attention_mask': attention_mask, # binary tensor that specifies which tokens should be attended to by the model
                'labels': labels # The loss is calculated by comparing the predicted token with labels[k]
            })

            # Debug outputs
            # torch.set_printoptions(profile="full")
            # print(f"Total length: {len(input_ids)}")
            # print(f"Padding tokens: {pad_length}")
            # print(f"Content tokens: {len(full_content)} (prompt+profile+eos)")
            # print(f"Mask spans: padding [0-{pad_length-1}], prompt [{pad_length}-{pad_length+self.prompt_length-1}]")
            # print("First 15 tokens:")
            # print(f"Input IDs: {input_ids[:15].tolist()}")
            # print(f"Labels:    {['-100' if x == -100 else x for x in labels[:15].tolist()]}")
            # print(f"Decoded content start: '{tokenizer.decode(input_ids[pad_length:pad_length+15])}'")

            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    

def prepare_dataset(profiles, tokenizer):
    return JobProfileDataset(profiles, tokenizer)

def train_model(model, dataset, tokenizer, output_dir="./lora_job_adapter"):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[GradientMonitorCallback(warning_threshold=0.3)]
    )
    
    trainer.train()
    model.save_pretrained(output_dir)


class GradientMonitorCallback(TrainerCallback):
    def __init__(self, warning_threshold=0.3):
        self.warning_threshold = warning_threshold
        self.last_warning_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        current_step = state.global_step
        grad_norm = logs.get('grad_norm', 0)
        
        # Only warn every 5 steps to avoid spam
        if grad_norm < self.warning_threshold: #and current_step > (self.last_warning_step + 5):
            print(f"\n⚠️ Step {current_step}: Low gradient norm ({grad_norm:.4f})")
            self.last_warning_step = current_step


def augment_profiles(profiles):
    import nlpaug.augmenter.char as nac
    
    # Custom tokenizers to preserve formatting
    def custom_tokenizer(text):
        return text.split(' ')  # Split on spaces only
    
    def custom_reverse_tokenizer(tokens):
        return ' '.join(tokens)  # Rejoin with single spaces
    
    # Initialize augmenter with format preservation
    aug = nac.KeyboardAug(
        aug_char_p=0.3,
        aug_word_p=0.3,
        tokenizer=custom_tokenizer,
        reverse_tokenizer=custom_reverse_tokenizer,
        include_special_char=False,
        stopwords=['*', '#', ':']  # Protect markdown syntax
    )
    
    augmented = []
    for profile in profiles:
        try:
            # Augment while preserving structure
            aug_text = aug.augment(profile)[0]
            
            # Post-process to fix any residual formatting issues
            aug_text = aug_text.replace(' * ', '*').replace(' : ', ':')
            augmented.append(aug_text)
        except Exception as e:
            print(f"Augmentation failed: {e}")
            augmented.append(profile)
            
    return augmented

def runMain():
    df = loadCSVData()
    profiles = generate_job_profiles_md(df)

    tokenizer = AutoTokenizer.from_pretrained(model_id,)
    print('special tokens are: ')
    print(tokenizer.special_tokens_map)
    
    tokenizer.add_eos_token = True
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # TRAIN LORA
    model = setup_model_for_training()
    model.resize_token_embeddings(len(tokenizer))  # Also need to resize embeddings for the model

    print('model is: ')
    print(model)

    dataset = prepare_dataset(profiles, tokenizer)    
    train_model(model, dataset, tokenizer)

    
if __name__ == "__main__":
    runMain()
    