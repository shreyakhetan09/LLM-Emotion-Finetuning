# LLM-Emotion-Finetuning
# 🧠 Behavioral Steering with LoRA — Fine-Tuning a 1.1B LLM on a Single GPU

Fine-tune a large language model to adopt a target personality using **Low Rank Adaptation (LoRA)** without touching a single original weight. This project demonstrates how Parameter-Efficient Fine-Tuning (PEFT) makes LLM customization accessible on consumer-grade hardware.

---

## 📌 Overview

Large Language Models like TinyLlama have over a billion parameters. Full fine-tuning is computationally prohibitive for most developers. This project uses **LoRA** to inject a small number of trainable adapter matrices into the model's attention layers, training only **~0.2% of total parameters** while achieving meaningful behavioral change.

As a concrete demonstration, the base model is steered from neutral/negative completions toward consistently **joyful, positive outputs** by training exclusively on emotion-labeled joy data.

| Prompt | Base Model | After LoRA |
|---|---|---|
| *"Emotionally, I am feeling"* | *"...going to be an absolute disaster..."* | *"...very positive and grateful..."* |

---

## 🔬 How LoRA Works

Standard fine tuning updates every weight matrix W in the network. LoRA instead **freezes W** and approximates the update as a product of two low rank matrices:

```
output = W_frozen(x) + (B @ A)(x) × (α / r)
```

- **A** has shape `(d × r)` and **B** has shape `(r × d)`, where rank `r << d`
- Only **A** and **B** are trained — drastically reducing memory and compute
- At inference, the adapter can be merged into W with zero added latency

This project targets the **Query (`q_proj`)** and **Value (`v_proj`)** projection matrices inside each transformer attention block, which most directly influence the model's generative behavior.

---

## 🗂️ Project Structure

```
.
├── lora_finetune.py       # Main training script
├── requirements.txt       # Dependencies
└── README.md
```

---

## ⚙️ Setup

**Requirements:** Python 3.9+, a CUDA-capable GPU (8GB+ VRAM recommended)

```bash
git clone https://github.com/your-username/lora-personality-finetuning
cd lora-personality-finetuning
pip install -r requirements.txt
```

**`requirements.txt`**
```
torch>=2.0.0
transformers>=4.38.0
peft>=0.9.0
datasets>=2.18.0
tqdm
```

---

## 🚀 Usage

```bash
python lora_finetune.py
```

The script will:
1. Load the **TinyLlama-1.1B** base model in float16 precision
2. Print a baseline completion for the test prompt
3. Load and filter the `dair-ai/emotion` dataset to joy-labeled samples
4. Configure and apply LoRA adapters
5. Train for 10 epochs and report loss per epoch
6. Print the fine-tuned model's completion for the same prompt

---

## 🧩 Configuration

Key hyperparameters are easy to adjust at the top of the script:

| Parameter | Value | Description |
|---|---|---|
| `r` | 16 | LoRA rank — higher = more expressive adapters, more params |
| `lora_alpha` | 32 | Scaling factor; effective scale = `alpha / r` |
| `target_modules` | `q_proj`, `v_proj` | Attention layers receiving adapters |
| `lora_dropout` | 0.05 | Regularization on adapter activations |
| `epochs` | 10 | Training epochs |
| `batch_size` | 8 | Samples per gradient step |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `max_length` | 64 | Max token length per training sample |

To steer toward a **different personality**, swap the dataset filter:

```python
# Current: joy (label == 1)
happy_dataset = dataset.filter(lambda x: x["label"] == 1)

# Example: sadness (label == 0), anger (label == 3), etc.
```
<img width="663" height="711" alt="image" src="https://github.com/user-attachments/assets/8ba26285-532b-4a11-9acc-4626f6f3c994" />

<img width="807" height="314" alt="image" src="https://github.com/user-attachments/assets/4afb53d4-6663-47e7-b630-d6fd5f47f1b1" />

---

## 📊 Results

Training converges steadily over 10 epochs on 1,000 joy samples:

```
Epoch 1  Average Loss: 4.5013
Epoch 2  Average Loss: 3.3264
Epoch 3  Average Loss: 3.1432
Epoch 4  Average Loss: 3.0240
Epoch 5  Average Loss: 2.9043
Epoch 6  Average Loss: 2.7897
Epoch 7  Average Loss: 2.6783
Epoch 8  Average Loss: 2.5746
Epoch 9  Average Loss: 2.4830
Epoch 10 Average Loss: 2.3901
```

Only **2,252,800 parameters** are trained out of **1,102,301,184** total — just **0.204%**.

---

## 🛠️ Tech Stack

- [PyTorch](https://pytorch.org/) — tensor computation and training loop
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — model and tokenizer loading
- [PEFT](https://huggingface.co/docs/peft) — LoRA implementation
- [Datasets](https://huggingface.co/docs/datasets) — emotion dataset loading and filtering
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b) — base language model
- [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — labeled emotion dataset
