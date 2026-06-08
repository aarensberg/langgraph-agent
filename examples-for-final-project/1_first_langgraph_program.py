from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define the state schema
class State(TypedDict):
    original_text: str
    upper_text: str
    length: int

# 2. Create the state graph
graph = StateGraph(State)

# 3. Define the node functions
def to_uppercase(state):
    text = state['original_text']
    return {"upper_text": text.upper()}

def count_characters(state):
    text = state['upper_text']
    return {"length": len(text)}

# 4. Add the nodes to the graph
graph.add_node("Upper", to_uppercase)
graph.add_node("Count", count_characters)

# 5. Connect the nodes in sequence
graph.add_edge(START, "Upper")
graph.add_edge("Upper", "Count")
graph.add_edge("Count", END)

# 6. Compile the graph
compiled_graph = graph.compile()

# 7. Invoke the graph with an initial state
initial_state = {"original_text": "Hello world"}
result = compiled_graph.invoke(initial_state)
print(result)
