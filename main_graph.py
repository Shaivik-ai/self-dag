from langgraph.graph import StateGraph, END
from nodes.inference_node import inference_node
from nodes.confidence_check_node import confidence_check_node
from nodes.fallback_node import fallback_node

builder = StateGraph(dict)

builder.add_node("inference", inference_node)
builder.add_node("confidence_check", confidence_check_node)
builder.add_node("fallback", fallback_node)

builder.set_entry_point("inference")
builder.add_edge("inference", "confidence_check")
builder.add_conditional_edges("confidence_check", lambda state: state["next"], {
    "end": END,
    "fallback": "fallback"
})
builder.add_edge("fallback", END)

graph = builder.compile()
