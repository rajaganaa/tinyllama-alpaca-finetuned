---
language: en
license: apache-2.0
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
- llm
- lora
- fine-tuned
- alpaca
---

# TinyLlama Alpaca Fine-tuned

Fine-tuned TinyLlama-1.1B on Alpaca dataset using LoRA.
Built from scratch as part of AI engineering learning journey.

## Training
- Base: TinyLlama-1.1B-Chat-v1.0
- Dataset: Alpaca (5000 samples)
- Method: LoRA (r=16, alpha=32)
- Epochs: 3 | Final loss: 1.01
- GPU: NVIDIA L4

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("RajGana/tinyllama-alpaca-finetuned")
model = AutoModelForCausalLM.from_pretrained("RajGana/tinyllama-alpaca-finetuned")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<|system|>You are a helpful assistant.</s><|user|>What is AI?</s><|assistant|>"
print(pipe(prompt, max_new_tokens=100)[0]["generated_text"])
```
