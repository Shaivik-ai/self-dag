# nodes/inference_node.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

model_path = "./fine_tuned_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

def inference_node(state):
    text = state["text"]
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    
    label = prediction.item()
    confidence = confidence.item()
    
    label_map = {0: "Negative", 1: "Positive"}
    final_label = label_map.get(label, f"LABEL_{label}")

    print(f"[InferenceNode] Predicted label: {final_label} | Confidence: {round(confidence * 100, 2)}%")

    return {
        **state,
        "predicted_label": final_label,
        "confidence": confidence,
        "next": "confidence_check"
    }
