from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import logging
from langgraph.graph import StateGraph, END

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model.eval()

logging.basicConfig(filename="logs/run_log.txt", level=logging.INFO)
