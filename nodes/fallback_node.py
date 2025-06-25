# nodes/fallback_node.py
def fallback_node(state):
    original = state["text"]
    print(f"[FallbackNode] Could you clarify your intent?\nWas this a negative review?")
    user_input = input("User: ").strip().lower()

    if "yes" in user_input or "negative" in user_input:
        final_label = "Negative"
    elif "no" in user_input or "positive" in user_input:
        final_label = "Positive"
    else:
        final_label = state["predicted_label"]  # fallback to model output

    return {
        **state,
        "final_label": final_label,
        "status": "corrected"
    }
