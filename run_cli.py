from main_graph import graph
from datetime import datetime
import json
import os

log_path = "logs/output_log.jsonl"
os.makedirs("logs", exist_ok=True)

print("✅ CLI loaded. Type 'exit' to quit.")

while True:
    text = input("\n📨 Enter input (or type 'exit'): ").strip()
    if text.lower() == "exit":
        break

    state = {"text": text}
    result = graph.invoke(state)

    log_data = {
        "timestamp": str(datetime.now()),
        "input": text,
        "output": result["final_label"],
        "status": result["status"],
        "confidence": result.get("confidence"),
        "fallback_triggered": result.get("fallback_triggered", False)
    }

    print(f"\n✅ Final Label: {log_data['output']} | Status: {log_data['status']}\n")

    with open(log_path, "a") as log_file:
        log_file.write(json.dumps(log_data) + "\n")
