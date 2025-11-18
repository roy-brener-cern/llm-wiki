import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pyarrow as pa # Needed for Arrow Table creation

print("--- Script Initialization Started (Python Native Batched) ---")

# --- Configuration ---
CLEAN_DIR = "./twiki_clean_text/"
OUTPUT_JSON = "synthetic_atlas_training_data.json"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHUNK_SIZE = 1800  # CRITICAL FIX: Reduced chunk size to fit within 2048 token limit
BATCH_SIZE = 8 # Stable batch size
# --- End Configuration ---

# 1. Initialize GPU/CPU and Model/Tokenizer
device = 0 if torch.cuda.is_available() else -1
print(f"CUDA availability check: {'GPU found' if device == 0 else 'Running on CPU'}")
print(f"Loading model '{MODEL_ID}'. This may take a minute...")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto" 
)
print(f"Model and Tokenizer loaded successfully.")

# Create a generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=True,
    max_new_tokens=512,
    temperature=0.7,
)

# --- Data Preparation ---
text_chunks_for_processing = []
clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.txt') or f.endswith('.md')]

print(f"Loading and chunking {len(clean_files)} cleaned files...")
for filename in tqdm(clean_files, desc="Chunking Files"):
    filepath = os.path.join(CLEAN_DIR, filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        continue

    # Simple chunking logic
    chunks = [content[i:i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
    text_chunks_for_processing.extend(chunks)

print(f"Total chunks created: {len(text_chunks_for_processing)}")

# --- Generation Logic ---

def create_prompt(text_chunk):
    """Generates the structured instruction prompt for the LLM."""
    
    system_instruction = (
        "You are a helpful data generator. Act as a student and a researcher. "
        "Read the provided text and generate exactly 4 realistic question-and-answer pairs "
        "based ONLY on the information in the text. The output MUST be a valid JSON array "
        "of objects with 'instruction' and 'response' keys. Do not include any text outside of the JSON."
    )
    
    user_prompt = f"TEXT:\n---\n{text_chunk}\n---\n\nGenerate 4 Q&A pairs based on the text above."
    
    prompt_messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    return tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)


# Convert all text chunks into formatted prompts
prompts_list = [create_prompt(chunk) for chunk in text_chunks_for_processing]

all_qa_pairs = []
count = 0
total_prompts = len(prompts_list)

print(f"Starting batched Q&A generation with BATCH SIZE {BATCH_SIZE}. Total prompts: {total_prompts}")

# Process the prompts in batches
for i in tqdm(range(0, total_prompts, BATCH_SIZE), desc="Batched GPU Inference"):
    batch_prompts = prompts_list[i:i + BATCH_SIZE]
    
    try:
        # Run the pipeline in batched mode!
        raw_outputs = generator(
            batch_prompts,
            batch_size=len(batch_prompts), # Use the actual batch size
            return_full_text=False,
            max_new_tokens=512,
            pad_to_multiple_of=None 
        )
    except Exception as e:
        # Log the error but continue to the next batch
        print(f"\n[ERROR] Skipping batch starting at index {i}. Reason: {e}")
        continue 

    # Process outputs
    for result in raw_outputs:
        output = result[0]['generated_text']
        
        try:
            if output.startswith("```json"):
                output = output.strip("```json").strip("```")

            qa_list = json.loads(output)
            
            if isinstance(qa_list, list):
                all_qa_pairs.extend([item for item in qa_list if 'instruction' in item and 'response' in item])
                count += len(qa_list)
                
        except (json.JSONDecodeError, IndexError):
            # Skip chunks where JSON parsing fails
            continue

# Final save
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(all_qa_pairs, f, indent=2)

print(f"\n--- Complete ---")
print(f"Total synthetic Q&A pairs generated: {count}")
print(f"Data saved to: {OUTPUT_JSON}")