# nodes/confidence_check_node.py
def confidence_check_node(state):
    confidence = state["confidence"]
    threshold = 0.8

    if confidence >= threshold:
        print("[ConfidenceCheckNode] Confidence sufficient. Accepting prediction.")
        return {
            **state,
            "status": "accepted",
            "final_label": state["predicted_label"],
            "next": "end",
            "fallback_triggered": False
        }
    else:
        print(f"[ConfidenceCheckNode] Confidence too low ({round(confidence * 100, 2)}%). Triggering fallback...")
        return {
            **state,
            "next": "fallback",
            "fallback_triggered": True
        }
