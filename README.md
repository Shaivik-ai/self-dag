
# 🧠 Self-Healing Classification DAG using LangGraph

This project introduces an intelligent, fault-tolerant classification system structured as a **DAG (Directed Acyclic Graph)** using **LangGraph** and a **fine-tuned transformer**. It's designed to recover from low-confidence predictions or failures by activating fallback logic automatically.

---

## 📌 Overview

Conventional NLP classification pipelines often fail silently when confidence scores are low or when primary models underperform. This DAG-based system takes a modular approach — each node handles a specific function like inference, confidence assessment, and fallback routing. If the primary classifier isn't confident, control is routed to a backup strategy.

---

## ⚙️ Key Components

### 🧩 Nodes

- **Inference Node:** Uses a fine-tuned transformer model to make predictions.
- **Confidence Check Node:** Evaluates prediction confidence; passes or reroutes accordingly.
- **Fallback Node:** Handles uncertain predictions using rule-based logic or alternate prompts.

### 📈 Logging

Structured logs (`output_log.jsonl`) are generated with details like:
- Input text
- Model prediction
- Confidence score
- Path taken through the DAG

---

## 🛠️ Tech Stack

- **LangGraph**
- **HuggingFace Transformers**
- **Pydantic** (for clean interfaces)
- **Click** (for CLI support)
- **Python 3.10+**

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Shaivik-ai/self-dag.git
   cd self-dag
````

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run via CLI:

   ```bash
   python run_cli.py --text "Your test input here"
   ```
````
## 📂 Directory Structure

```
self-dag/
│
├── nodes/                  # DAG Nodes
│   ├── inference_node.py
│   ├── confidence_check_node.py
│   └── fallback_node.py
│
├── model/                  # Fine-tuned transformer checkpoints
├── fine_tuned_model/       # Adapter files
├── logs/                   # Output logs
├── main_graph.py           # DAG Definition
├── run_cli.py              # CLI Interface
├── train_sentiment.py      # Model training script
├── dag_wokflow.py          # DAG Workflow wrapper
└── README.md               # You're here
```

---

## 🧠 Why This Matters

This project shows how **workflow-based AI architectures** can adapt dynamically to uncertain scenarios. It’s ideal for:

* Customer support automation
* Sentiment classification
* Medical triage systems
* Any system needing **high-reliability NLP**

---

## 🙌 Authors & Credits

* Built and tested by contributors at **Shaivik.ai**
* Fine-tuned using custom sentiment dataset
* Inspired by dynamic agent architectures

---

## 📜 License

MIT License — Feel free to use, adapt, and share.

---

## 📬 Contact

For collaborations or queries, reach out to the team at [Shaivik.ai](https://github.com/Shaivik-ai).



