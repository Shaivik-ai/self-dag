# train_sentiment.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os

# Load dataset
dataset = load_dataset("imdb")

# Use only labeled data
dataset = {
    "train": dataset["train"].shuffle(seed=42),
    "test": dataset["test"]
}


# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Apply LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # FIXED HERE
)

model = get_peft_model(base_model, peft_config)


# Tokenization
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

# Encode separately
encoded_train = dataset["train"].map(preprocess, batched=True)
encoded_test = dataset["test"].map(preprocess, batched=True)

# Format for PyTorch
encoded_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
encoded_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Training Args
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True  # ⚡️ Use GPU half precision for speed (optional)
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
