from ..config.settings import setup_langsmith, check_api_keys
from ..tools.tools import query_invoice_history, get_recommendations, request_refund, process_purchase
from ..tools.tools import update_profile, fetch_customer_info, parse_track_selection, parse_track_selection
from ..utils.helpers import create_tool_node_with_fallback
from ..utils.database import initialize_vector_store

from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition
from langchain import hub


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # Sometimes the LLM returns empty/error responses, simply try a quick retry.
            
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, dict) and result.content.get("status") == "error"
                or isinstance(result.content, list) and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Please try a different approach or provide more information.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Need API keys to run this thing
check_api_keys()

# Setup tracing
client = setup_langsmith()

# Initialize vector store - needed for recommendations and other operations
# This ensures it's available when running in Langgraph Studio
try:
    initialize_vector_store()
except Exception as e:
    print(f"Warning: Failed to initialize vector store: {str(e)}")
    print("Some recommendations features may not work properly.")

# Load up the brain
llm = ChatOpenAI(temperature=0.0, model="gpt-4o")

# Load prompt from langchain hub
prompt = hub.pull("music-store-agent-prompt")

# Split tools into ones that need approval and ones that don't
safe_tools = [get_recommendations, query_invoice_history, fetch_customer_info, parse_track_selection]
sensitive_tools = [process_purchase, request_refund, update_profile]

# Quick lookup for tool names
safe_tool_names = {t.name for t in safe_tools}
sensitive_tool_names = {t.name for t in sensitive_tools}

# Hook up the prompt to the LLM with all tools attached
assistant_runnable = prompt | llm.bind_tools(safe_tools + sensitive_tools)

# Start building our graph
builder = StateGraph(State)

# Add all the nodes we need
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_fallback(safe_tools))
builder.add_node(
    "sensitive_tools", create_tool_node_with_fallback(sensitive_tools)
)

# Start with the assistant
builder.add_edge(START, "assistant")

# Route to the right node based on if tool needs approval
def route_tools(state: State):
    next_node = tools_condition(state)
    # No tools? Just go back to the user
    if next_node == END:
        return END
    
    ai_message = state["messages"][-1]
    # We only handle one tool at a time for now
    # Would need ANY condition for parallel tools
    first_tool_call = ai_message.tool_calls[0]
    
    # Route to the right node based on if tool needs approval
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"

# Hook up all the edges between nodes
builder.add_conditional_edges(
    "assistant", route_tools, ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

# Set up memory to keep track of state
memory = MemorySaver()
# Compile the whole graph, adding interrupt points
graph = builder.compile(
    checkpointer=memory,
    # Stop and ask for permission before doing sensitive stuff
    # User can review/approve/reject at this point
    interrupt_before=["sensitive_tools"],
)

