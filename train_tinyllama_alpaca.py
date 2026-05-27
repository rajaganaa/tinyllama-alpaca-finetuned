"""
TinyLlama-1.1B Fine-tuning on Alpaca Dataset using LoRA
========================================================
Base Model : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Dataset    : Alpaca (5000 samples)
Method     : LoRA (r=16, alpha=32)
Epochs     : 3  |  Final loss ~1.01
GPU        : NVIDIA L4 (GCP)
Author     : RajGana
HF Repo    : RajGana/tinyllama-alpaca-finetuned
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR   = "./tinyllama-alpaca-finetuned"
HF_REPO      = "RajGana/tinyllama-alpaca-finetuned"

MAX_LENGTH   = 512
NUM_SAMPLES  = 5000
EPOCHS       = 3
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
LR           = 2e-4

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET  = ["q_proj", "v_proj", "k_proj", "o_proj"]


# ──────────────────────────────────────────────
# 1. Load Tokenizer
# ──────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"


# ──────────────────────────────────────────────
# 2. Load Model with 4-bit Quantisation (QLoRA)
# ──────────────────────────────────────────────
print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)


# ──────────────────────────────────────────────
# 3. Apply LoRA
# ──────────────────────────────────────────────
print("Applying LoRA...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ──────────────────────────────────────────────
# 4. Dataset – Alpaca format
# ──────────────────────────────────────────────
def format_alpaca_prompt(sample):
    """Convert Alpaca sample to TinyLlama chat format."""
    instruction = sample["instruction"]
    input_text  = sample.get("input", "")
    output      = sample["output"]

    if input_text:
        user_msg = f"{instruction}\n{input_text}"
    else:
        user_msg = instruction

    prompt = (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{user_msg}</s>\n"
        f"<|assistant|>\n{output}</s>"
    )
    return {"text": prompt}


print("Loading dataset...")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES))
dataset = dataset.map(format_alpaca_prompt, remove_columns=dataset.column_names)


def tokenize(sample):
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


print("Tokenising...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.train_test_split(test_size=0.1, seed=42)

train_dataset = tokenized["train"]
eval_dataset  = tokenized["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")


# ──────────────────────────────────────────────
# 5. Training Arguments
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    push_to_hub=False,
    optim="paged_adamw_8bit",
)


# ──────────────────────────────────────────────
# 6. Trainer
# ──────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()


# ──────────────────────────────────────────────
# 7. Save & Push
# ──────────────────────────────────────────────
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
print(f"Push to HF: huggingface-cli upload {HF_REPO} {OUTPUT_DIR}")


# ──────────────────────────────────────────────
# 8. Quick Inference Test
# ──────────────────────────────────────────────
def generate(prompt, max_new_tokens=200):
    from transformers import pipeline
    pipe = pipeline("text-generation", model=OUTPUT_DIR, tokenizer=OUTPUT_DIR,
                    torch_dtype=torch.bfloat16, device_map="auto")
    full_prompt = (
        f"<|system|>\nYou are a helpful assistant.</s>\n"
        f"<|user|>\n{prompt}</s>\n"
        f"<|assistant|>\n"
    )
    out = pipe(full_prompt, max_new_tokens=max_new_tokens, do_sample=True,
               temperature=0.7, top_p=0.9)
    return out[0]["generated_text"]


if __name__ == "__main__":
    print("\n--- Inference Test ---")
    print(generate("What is machine learning?"))
