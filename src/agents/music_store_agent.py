from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
import uuid
import logging
import json
from ..config.settings import setup_langsmith, check_api_keys
from ..utils.database import setup_vector_store
from ..tools.music_tools import query_invoice_history, get_recommendations, update_memory

# Configure logging
logger = logging.getLogger(__name__)

# Check for API keys
check_api_keys()

# Initialize LangSmith
client = setup_langsmith()

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = setup_vector_store(embeddings)

# Initialize the LLM
llm = ChatOpenAI(temperature=0.1, model="gpt-4-turbo-preview")
logger.info("LLM initialized")

# System prompts
SYSTEM_PROMPT = """You are a helpful music store assistant that can help users with music recommendations and invoice queries.
You have access to tools to:
1. Query customer invoice history
2. Get music recommendations
3. Update customer profile information

Here is the current Customer Profile (may be empty if no information collected yet):
<customer_profile>
{customer_profile}
</customer_profile>

Your task is to:
1. Help customers find music they might enjoy
2. Look up their purchase history when requested
3. Maintain their profile with music preferences and habits

When responding:
1. Be conversational and friendly
2. Use the tools available to gather information
3. Update the customer profile when new preferences are learned
4. Make personalized recommendations based on their profile

Remember to update the customer profile when you learn new information about their music tastes or preferences."""

def task_agent(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main agent that processes messages and decides on actions."""
    try:
        # Get the customer ID from config
        customer_id = config["configurable"]["customer_id"]
        
        # Get existing profile from store
        namespace = ("profile", customer_id)
        profile = store.get(namespace, "profile")
        profile_content = profile.value if profile else CustomerProfile().model_dump()
        
        # Format system message with context
        system_msg = SYSTEM_PROMPT.format(customer_profile=profile_content)
        
        # Create messages list with system prompt
        messages = [SystemMessage(content=system_msg)] + state["messages"]
        
        # Get response from model with tools
        model = llm.bind(
            tools=[query_invoice_history, get_recommendations, update_memory],
            tool_choice="auto"
        )
        response = model.invoke(messages)
        
        return {"messages": state["messages"] + [response]}
    except Exception as e:
        logger.error(f"Error in task_agent: {str(e)}", exc_info=True)
        return {"messages": state["messages"] + [AIMessage(content=f"I encountered an error: {str(e)}")]}

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_profile", "process_tools"]:
    """Route the message based on the agent's response."""
    message = state['messages'][-1]
    
    if not message.tool_calls:
        return END
        
    tool_call = message.tool_calls[0]
    
    try:
        args = json.loads(tool_call["function"]["arguments"])
        if tool_call["name"] == "UpdateMemory":
            update_type = args.get('update_type')
            if update_type == "profile":
                return "update_profile"
        return "process_tools"
    except Exception as e:
        logger.error(f"Error in route_message: {str(e)}")
        return "process_tools"

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Update the customer profile with new information."""
    try:
        customer_id = config["configurable"]["customer_id"]
        namespace = ("profile", customer_id)
        
        # Get existing profile
        profile = store.get(namespace, "profile")
        existing_profile = profile.value if profile else CustomerProfile().model_dump()
        
        # Update profile using the model
        system_msg = f"""Based on the conversation, extract any new information about the customer's music preferences.
Current profile: {existing_profile}

Output should be a valid JSON object matching the CustomerProfile schema with these fields:
- favorite_genres: list[str]
- favorite_artists: list[str]
- recent_purchases: list[str]
- listening_preferences: list[str]

Only include information that appears in the conversation."""
        
        # Get profile updates from LLM
        profile_update = llm.invoke(
            [SystemMessage(content=system_msg)] + state["messages"]
        )
        
        try:
            # Parse LLM response into CustomerProfile
            profile_data = json.loads(profile_update.content)
            updated_profile = CustomerProfile(**profile_data)
            
            # Merge with existing profile
            merged_profile = CustomerProfile(
                favorite_genres=list(set(existing_profile.get('favorite_genres', []) + updated_profile.favorite_genres)),
                favorite_artists=list(set(existing_profile.get('favorite_artists', []) + updated_profile.favorite_artists)),
                recent_purchases=list(set(existing_profile.get('recent_purchases', []) + updated_profile.recent_purchases)),
                listening_preferences=list(set(existing_profile.get('listening_preferences', []) + updated_profile.listening_preferences))
            )
            
            # Store updated profile
            store.put(namespace, "profile", merged_profile.model_dump())
            
            return {"messages": state["messages"] + [AIMessage(content="Profile updated successfully")]}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM profile update: {str(e)}")
            return {"messages": state["messages"] + [AIMessage(content="Failed to update profile: Invalid format")]}
            
    except Exception as e:
        logger.error(f"Error in update_profile: {str(e)}", exc_info=True)
        return {"messages": state["messages"] + [AIMessage(content=f"Error updating profile: {str(e)}")]}

def process_tools(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Process any tool calls from the agent."""
    logger.info("Processing tool calls")
    message = state['messages'][-1]
    customer_id = config["configurable"]["customer_id"]
    
    results = []
    for tool_call in message.tool_calls:
        try:
            # Parse arguments from the tool call
            if isinstance(tool_call["function"]["arguments"], str):
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse arguments: {tool_call['function']['arguments']}")
                    args = {}
            else:
                args = tool_call["function"]["arguments"]
            
            if tool_call["name"] == "query_invoice_history":
                result = query_invoice_history.invoke(customer_id)
            elif tool_call["name"] == "get_recommendations":
                # Extract genre from arguments
                genre = args.get('genre')
                if not genre:
                    logger.warning("No genre provided in arguments")
                    result = "Error: No genre provided in the request"
                else:
                    result = get_recommendations.invoke(genre, vector_store)
            elif tool_call["name"] == "UpdateMemory":
                logger.info("Processing memory update")
                continue
                
            results.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call["id"]
            })
        except Exception as e:
            logger.error(f"Error processing tool call: {str(e)}", exc_info=True)
            results.append({
                "role": "tool",
                "content": f"Error processing tool call: {str(e)}",
                "tool_call_id": tool_call["id"]
            })
    
    return {"messages": results}

# Create the graph
def create_graph():
    """Create the agent workflow graph."""
    # Initialize graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("task_mAIstro", task_agent)
    workflow.add_node("update_profile", update_profile)
    workflow.add_node("process_tools", process_tools)
    
    # Add edges with conditional routing
    workflow.add_edge(START, "task_mAIstro")
    workflow.add_conditional_edges(
        "task_mAIstro",
        route_message,
        {
            "update_profile": "update_profile",
            "process_tools": "process_tools",
            END: END
        }
    )
    workflow.add_edge("update_profile", "task_mAIstro")
    workflow.add_edge("process_tools", "task_mAIstro")
    
    # Initialize memory systems
    memory_store = InMemoryStore()
    
    # Compile and return the graph
    return workflow.compile(
        checkpointer=MemorySaver(),
        store=memory_store
    )

# Create graph instance
graph = create_graph()

def get_agent_response(message: str, customer_id: str, thread_id: str = None):
    """Get a response from the agent for a specific customer."""
    logger.info("Starting new conversation turn")
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    config = {
        "configurable": {
            "customer_id": customer_id,
            "thread_id": thread_id
        }
    }
    
    # Create initial state
    messages = [HumanMessage(content=message)]
    
    try:
        # Run the graph
        result = graph.invoke({"messages": messages}, config)
        return result["messages"], thread_id
    except Exception as e:
        logger.error(f"Error in conversation: {str(e)}", exc_info=True)
        return [AIMessage(content="I apologize, but I encountered an error while processing your request. Please try again.")], thread_id 