# 🦙 TinyLlama Alpaca Fine-tuned

> Fine-tuned **TinyLlama-1.1B-Chat** on the Stanford Alpaca dataset using **LoRA (Low-Rank Adaptation)** — achieving instruction-following capability at a fraction of the compute cost of full fine-tuning.

[![HuggingFace](https://img.shields.io/badge/🤗_Model-RajGana/tinyllama--alpaca--finetuned-yellow)](https://huggingface.co/RajGana/tinyllama-alpaca-finetuned)
[![Downloads](https://img.shields.io/badge/Downloads-33+-brightgreen)](https://huggingface.co/RajGana/tinyllama-alpaca-finetuned)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

---

## 🧠 What Is This?

This project fine-tunes **TinyLlama-1.1B** — a small but capable LLM — on the Alpaca instruction dataset using **LoRA**, enabling:

- Instruction-following without full model fine-tuning
- Training on a single GPU (NVIDIA L4) in under an hour
- A deployable 1B model that responds like a helpful assistant

LoRA freezes the base model and injects **trainable low-rank matrices** into attention layers, reducing trainable parameters by ~99% while preserving model quality.

---

## 📋 Training Details

| Setting | Value |
|---------|-------|
| Base Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Dataset | Stanford Alpaca — **5,000 samples** |
| Method | **LoRA** (r=16, alpha=32) |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Final Loss | **1.01** |
| Precision | bfloat16 |
| GPU | NVIDIA L4 (GCP Workbench) |
| Optimizer | AdamW (paged, 8-bit) |

---

## 🚀 Quick Start

### Install Dependencies
```bash
git clone https://github.com/rajaganaa/tinyllama-alpaca-finetuned.git
cd tinyllama-alpaca-finetuned
pip install -r requirements_2.txt
```

### Run Inference
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("RajGana/tinyllama-alpaca-finetuned")
model = AutoModelForCausalLM.from_pretrained("RajGana/tinyllama-alpaca-finetuned")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is machine learning?</s>\n<|assistant|>\n"
print(pipe(prompt, max_new_tokens=200)[0]["generated_text"])
```

### Train Your Own
```bash
python train_tinyllama_alpaca.py
```

---

## 📁 Repository Structure

```
tinyllama-alpaca-finetuned/
├── train_tinyllama_alpaca.py  # LoRA fine-tuning script
├── requirements_2.txt         # Python dependencies
├── config.json                # Model architecture config
├── tokenizer.json             # Tokenizer vocabulary
├── tokenizer_config.json      # Tokenizer settings
├── chat_template.jinja        # Chat prompt template
├── generation_config.json     # Inference generation settings
└── README.md
```

---

## 💡 Key Learnings

- How LoRA injects trainable rank decomposition matrices without modifying base weights
- Why `paged_adamw_8bit` reduces GPU memory significantly during training
- The impact of prompt formatting on instruction-following quality
- How a 1.1B model can match much larger models on focused tasks with good fine-tuning

---

## 📊 Model Performance

The model achieves a final training loss of **1.01** after 3 epochs on 5,000 Alpaca samples — comparable to published LoRA fine-tuning results on similar datasets. It has received **33+ downloads** on Hugging Face within days of release.

---

## 🔗 Related Work

- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) — instruction-following dataset
- [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) — base model
- [Hugging Face Model](https://huggingface.co/RajGana/tinyllama-alpaca-finetuned) — weights hosted here

---

## 👨‍💻 Author

**Rajaganapathy M**  
AI/ML Engineer | M.Tech AI | SRM Institute of Science & Technology  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/raja-ganapathy-36b00658)
[![GitHub](https://img.shields.io/badge/GitHub-rajaganaa-black?logo=github)](https://github.com/rajaganaa)
[![HuggingFace](https://img.shields.io/badge/🤗-RajGana-yellow)](https://huggingface.co/RajGana)
