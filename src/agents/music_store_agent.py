from ..config.settings import setup_langsmith, check_api_keys
from ..tools.tools import query_invoice_history, get_recommendations, request_refund, process_purchase
from ..tools.tools import update_profile, fetch_customer_info, parse_track_selection, parse_track_selection
from ..utils.helpers import create_tool_node_with_fallback

from typing import Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import tools_condition


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_info: dict  # Customer data cache
    selected_tracks: list[int] | None  # Tracks user wants to buy

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # Sometimes the LLM returns empty/error responses
            # Just kick it to try again in those cases
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

# Load up the brain
llm = ChatOpenAI(temperature=0.0, model="gpt-4o")

assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful music store assistant for an online digital music store. You help customers discover, 
purchase, and manage their music collection.

1. Music Discovery and Recommendations:
   - Search for music based on genres, moods, styles, or similarity to other songs using the get_recommendations tool
   - Show detailed track information including artists, albums, and prices
   - Customers DO NOT have logged preferences so if you need clarification on what they want, ask them to specify their preferences

2. Purchase Management:
   - Parse track selections from recommendations (by track numbers or names) using the parse_track_selection tool
   - Preview purchase details before confirmation using the process_purchase tool
   - Process music track purchases after user confirmation
   - Show complete purchase history with the query_invoice_history tool 
   - Handle refund requests for previous purchases using the request_refund tool

3. Profile Management:
    - Get customer information using the fetch_customer_info tool.
   - Update customer contact information using the update_profile tool
   - Modify billing/shipping addresses using the update_profile tool
   - Update customer details using the update_profile tool

When assisting customers:
- Only process requests from the customer who is currently logged in
- For purchases, always show a preview first before completing the transaction
- When recommending music, be creative and consider various aspects (genre, mood, style) depending on the query.
- If a search returns no results, try broadening the search criteria
- Provide clear explanations for any errors or issues
- For purchases from recommendations, help users select tracks by number (e.g., "tracks 1, 3, and 5") or by name

Remember to be conversational and helpful while ensuring secure and accurate transactions.
"""
        ),
        (
            "placeholder",
            "{messages}"
        )
    ]
).partial(time=datetime.now)


# Split tools into ones that need approval and ones that don't
safe_tools = [get_recommendations, query_invoice_history, fetch_customer_info, parse_track_selection]
sensitive_tools = [process_purchase, request_refund, update_profile]

# Quick lookup for tool names
safe_tool_names = {t.name for t in safe_tools}
sensitive_tool_names = {t.name for t in sensitive_tools}

# Hook up the prompt to the LLM with all tools attached
assistant_runnable = assistant_prompt | llm.bind_tools(safe_tools + sensitive_tools)

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

# Usage examples (commented out):
# 1. Run the graph until it hits a sensitive tool
# thread = {"configurable": {"thread_id": "unique_thread_id"}}
# for event in graph.stream(inputs, thread, stream_mode="values"):
#     print(event)
#
# 2. User approves - continue with no changes
# for event in graph.stream(None, thread, stream_mode="values"):
#     print(event)
#
# 3. Edit the tool call before approving
# graph.update_state(thread, {"messages": [edited_messages]})
# for event in graph.stream(None, thread, stream_mode="values"):
#     print(event)
#
# 4. User cancels
# graph.update_state(thread, {"messages": state["messages"] + [("user", "I've decided to cancel this operation.")]})
# for event in graph.stream(None, thread, stream_mode="values"):
#     print(event)

