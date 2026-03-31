"""
Behavioral Steering with LoRA (Making the Model Happy!)
Fine-tunes TinyLlama-1.1B on joyful text using Low-Rank Adaptation (LoRA).
Only ~0.2% of parameters are trained, making this feasible on a single GPU.
"""

# ── 0. Install dependencies ───────────────────────────────────────────────────
# !pip install -q transformers peft datasets

# ── 1. Imports ────────────────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2. Load Tokenizer & Base Model ────────────────────────────────────────────
model_name = "TinyLlama/TinyLlama-1.1B-step-50K-105b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Needed so padding doesn't crash

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # Automatically places layers on available GPU(s)
    torch_dtype=torch.float16  # Half precision: halves VRAM vs float32
)

# ── 3. Task 4.1 — Baseline: See what the model says BEFORE fine-tuning ────────
neutral_prompt = "Emotionally, I am feeling"
inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    base_outputs = base_model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )
    print("─── Base Model Completion ───")
    print(tokenizer.decode(base_outputs[0], skip_special_tokens=True))
    # Expected: something neutral or negative, e.g. "...going to be an absolute disaster"

# ── 4. Task 4.2a — Load & Filter Dataset (joy label == 1) ────────────────────
dataset = load_dataset("dair-ai/emotion", "split", split="train")
happy_dataset = dataset.filter(lambda x: x["label"] == 1).select(range(1000))
# We only expose the model to joyful text so it steers toward positive generation

def tokenize_and_format(example):
    """
    Tokenize text and set labels = input_ids.
    For causal LM training, the model predicts the NEXT token at each position,
    so labels are just the inputs shifted by 1 (handled internally by HuggingFace).
    """
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=64,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_happy_dataset = happy_dataset.map(tokenize_and_format, batched=True)
tokenized_happy_dataset.set_format("torch")

# ── 5. Task 4.2b — Configure LoRA ────────────────────────────────────────────
# LoRA freezes base weights and injects trainable low-rank matrices A and B
# into the attention layers such that:  ΔW ≈ B @ A  (rank r << d)
#
# For each targeted layer: output = W_frozen(x) + (B @ A)(x) * (alpha / r)
#
# Only A and B are trained — ~2.25M params vs 1.1B total (0.2%)

lora_config = LoraConfig(
    r=16,                              # Rank: bottleneck size of adapter matrices
    lora_alpha=32,                     # Scaling factor; effective scale = alpha/r = 2.0
    target_modules=["q_proj", "v_proj"],  # Inject into Query & Value attention projections
    lora_dropout=0.05,                 # Dropout on adapter activations (regularization)
    bias="none",                       # Don't train bias terms
    task_type="CAUSAL_LM"             # Tells PEFT this is a generative language model
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 2,252,800 || all params: 1,102,301,184 || trainable%: 0.204

# ── 6. Task 4.3 — Training Loop ───────────────────────────────────────────────
train_dataloader = DataLoader(tokenized_happy_dataset, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-3)

epochs = 10
peft_model.train()

for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        # Move batch to GPU
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Standard 5-step PyTorch training loop:
        optimizer.zero_grad()                                        # 1. Zero gradients
        outputs = peft_model(                                        # 2. Forward pass
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss                                          # 3. Get loss
        loss.backward()                                              # 4. Backward pass
        optimizer.step()                                             # 5. Update weights
                                                                     #    (only A and B!)

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_dataloader):.4f}")

# ── 7. Task 4.4 — Evaluate: See the model's new "happy" personality ───────────
peft_model.eval()

with torch.no_grad():
    happy_outputs = peft_model.generate(
        **inputs,             # Same prompt as before: "Emotionally, I am feeling"
        max_new_tokens=40,
        do_sample=True,
        temperature=0.7
    )
    print("\n─── Happy LoRA Model Completion ───")
    print(tokenizer.decode(happy_outputs[0], skip_special_tokens=True))
    # Expected: something positive, e.g. "...very positive about this..."
