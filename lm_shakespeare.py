"""
Language Modeling with Transformers

Goal:
- Understand the workflow behind training causal language models

What this explores:
- Tokenization and batching for autoregressive models
- Fine-tuning GPT-2 on a small text corpus
- Generating text from a trained language model
"""

!pip install transformers
!pip install datasets
from datasets import load_dataset

ds = load_dataset('Trelis/tiny-shakespeare')
ds
DatasetDict({
    train: Dataset({
        features: ['Text'],
        num_rows: 472
    })
    test: Dataset({
        features: ['Text'],
        num_rows: 49
    })
})

print(ds['train']['Text'][0][:300])

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    return tokenizer(example["Text"], truncation=False)

tokenized_ds = ds.map(tokenize_fn, batched=True, remove_columns=["Text"])

from itertools import chain

block_size = 128

def group_texts(examples):
    concatenated_ids = sum(examples["input_ids"], [])
    concatenated_mask = sum(examples["attention_mask"], [])

    total_len = (len(concatenated_ids) // block_size) * block_size

    input_ids = [
        concatenated_ids[i:i + block_size]
        for i in range(0, total_len, block_size)
    ]
    attention_mask = [
        concatenated_mask[i:i + block_size]
        for i in range(0, total_len, block_size)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),
    }
lm_ds = tokenized_ds.map(group_texts, batched=True)

from torch.utils.data import DataLoader
from transformers import default_data_collator

train_dl = DataLoader(
    lm_ds["train"],
    batch_size=2,
    shuffle=True,
    collate_fn=default_data_collator,
)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for step, batch in enumerate(train_dl):
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.4f}")

    if step == 300:  # stop early for Colab sanity
        break

model.eval()

prompt = "ROMEO:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

out = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
