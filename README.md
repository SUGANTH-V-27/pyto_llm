# LLM — Transformer Language Model Built from Scratch

Milo is a GPT-style transformer language model implemented and trained from scratch using PyTorch.  
The goal of this project is to deeply understand how large language models work at both architectural and training system levels.

---

## 🚀 Overview

- Built a complete transformer-based language model without using high-level training frameworks
- Designed and trained models at **tens to hundreds of millions of parameters**
- Trained on **billions of tokens** using a custom distributed training pipeline
- Focused on understanding **training dynamics, scaling behavior, and system constraints**

---

## 🧠 Model Architecture

- GPT-style autoregressive transformer
- Multi-head self-attention
- Residual connections
- **RMSNorm (instead of LayerNorm)**
- **RoPE (Rotary Positional Embeddings)**
- **SwiGLU-style MLP layers**
- Weight tying between embedding and output layer

---

## ⚙️ Training Setup

| Component              | Details |
|----------------------|--------|
| Framework            | PyTorch |
| Sequence Length      | 1024 |
| Tokenizer            | GPT-2 / BPE |
| Optimizer            | AdamW |
| Loss                 | Cross-Entropy |
| LR Schedule          | Cosine Decay + Warmup |
| Precision            | Mixed Precision (AMP) |
| Gradient Accumulation| Yes |
| Gradient Clipping    | Yes |

---

## 📊 Models

### Model 1
- Parameters: **63.5M**
- Training Tokens: **1.9B**

### Model 2
- Parameters: **227M**
- Training Tokens: **2B+ (target: 4–5B)**

---

## 📚 Dataset

Streaming dataset pipeline with probabilistic mixing:

- FineWebEdu (~1.7B tokens)
- FineWeb
- Wikipedia
- Gutenberg books

**Mixing strategy:**
- 70% FineWeb
- 20% Wikipedia
- 10% Books

---

## ⚡ Training System

- Distributed training using **DDP (Distributed Data Parallel)**
- Multi-GPU setup: **2× T4 GPUs (16GB)**
- Communication backend: **NCCL**
- Gradient checkpointing for memory efficiency
- Custom streaming dataloader

**Throughput:**
- ~10,000 tokens/sec  
- ~596M tokens trained in 12 hours  

---

## 🔍 Inference

Implemented custom text generation with:

- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling
- Repetition penalty

---

## 🧪 Example Outputs
## 🧪 Evaluation Results

The model was evaluated at two stages:

- **Stage 1:** 1.7B tokens (FineWebEdu only)  
- **Stage 2:** 2B+ tokens (FineWeb + Wikipedia + Gutenberg mixture)

---

## 📊 Stage 1 — FineWebEdu (1.7B tokens)

### Example: Structured Continuation

**Prompt:**  
The capital of Japan is a city that  

**Output:**  
The capital of Japan is a city that has been ordered by the British since World War II... :contentReference[oaicite:0]{index=0}  

**Observation:**
- Fluent sentence structure  
- Strong grammatical coherence  
- Major factual errors (hallucinated historical details)  

---

### Example: Scientific Concept

**Prompt:**  
Photosynthesis is a process in which  

**Output:**  
Photosynthesis is a process in which plants have evolved from a different species of algae... :contentReference[oaicite:1]{index=1}  

**Observation:**
- Maintains structure of explanation  
- Produces incorrect scientific reasoning  
- Shows lack of grounded knowledge  

---

### Example: General Coherence

**Prompt:**  
Neural networks are systems that learn patterns from data by  

**Output:**  
Neural networks are systems that learn patterns from data by analyzing patterns and relationships... :contentReference[oaicite:2]{index=2}  

**Observation:**
- Good high-level explanation  
- Repetition and shallow reasoning  

---

### Summary (Stage 1)

**Strengths:**
- Strong fluency and grammar  
- Coherent sentence continuation  
- Good general language modeling  

**Limitations:**
- High factual inaccuracy  
- Weak reasoning (math/science)  
- Repetition and drift  

---

## 📊 Stage 2 — Mixed Dataset (2B+ tokens)

### Example: Structured Continuation

**Prompt:**  
The capital of Japan is a city that  

**Output:**  
The capital of Japan is a city that has been under the care of its inhabitants... :contentReference[oaicite:3]{index=3}  

**Observation:**
- Improved fluency  
- Less extreme hallucination  
- Still lacks precise factual grounding  

---

### Example: Scientific Concept

**Prompt:**  
Photosynthesis is a process in which  

**Output:**  
Photosynthesis is a process in which plants produce nitrogen (N) and oxygen (O)... :contentReference[oaicite:4]{index=4}  

**Observation:**
- Better structured explanation  
- Still factually incorrect  
- Shows partial improvement in reasoning structure  

---

### Example: Paragraph Coherence

**Prompt:**  
Artificial intelligence is transforming industries by  

**Output:**  
Artificial intelligence is transforming industries by offering new possibilities... :contentReference[oaicite:5]{index=5}  

**Observation:**
- Improved multi-sentence coherence  
- Better topic continuity  

---

## 📈 Comparison

| Capability            | Stage 1 (1.7B) | Stage 2 (2B+) |
|---------------------|--------------|--------------|
| Fluency             | Good         | Better       |
| Coherence           | Moderate     | Improved     |
| Factual Accuracy    | Poor         | Slightly better |
| Reasoning           | Weak         | Weak         |
| Stability           | Low          | Improved     |

---

## 🧠 Key Insight

The model learns:

- **Language structure first (grammar, flow)**
- **Then partial reasoning patterns**
- **Factual accuracy improves slowly and requires scale + better data**

Even with billions of tokens, the model:
- Can generate fluent text  
- But does not reliably produce correct answers  

---

## 🎯 Next Steps

- Scale training to **4–5B tokens**
- Introduce **math and code datasets**
- Improve evaluation benchmarks
- Reduce hallucination and drift
