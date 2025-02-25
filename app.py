import streamlit as st
import sqlite3
from typing import List, Tuple
from src.agents.music_store_agent import graph  # Our compiled graph with interrupt capability
from src.config.settings import setup_langsmith, check_api_keys
from src.utils.database import initialize_vector_store
import uuid
from langgraph.types import Command

# App config
st.set_page_config(
    page_title="Music Store Agent",
    page_icon="🎵",
    layout="wide"
)

# Make sure API keys exist before we do anything
check_api_keys()

# Setup LangSmith (for tracing)
client = setup_langsmith()

# Load up the vector store - needed for music recommendations
with st.spinner("Initializing vector store..."):
    initialize_vector_store()
    st.success("✅ Vector store ready")

# Set up session vars if they don't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

def get_customers() -> List[Tuple[int, str]]:
    """Grab the top 10 customers from DB, ordered by spending."""
    conn = sqlite3.connect('chinook.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            c.CustomerId,
            c.FirstName || ' ' || c.LastName as Name,
            COUNT(i.InvoiceId) as PurchaseCount,
            ROUND(SUM(i.Total), 2) as TotalSpent
        FROM Customer c
        LEFT JOIN Invoice i ON c.CustomerId = i.CustomerId
        GROUP BY c.CustomerId
        ORDER BY TotalSpent DESC
        LIMIT 10
    """)
    customers = cursor.fetchall()
    conn.close()
    return customers

# Extract tool call info for the approval UI
def extract_tool_call_info(state):
    """Pull out the pending tool info from state for display."""
    if not state or "messages" not in state:
        return None
    
    # Look for the last message with a tool call
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]  # Just grab the first one
            return {
                "name": tool_call.get("name", "Unknown tool"),
                "args": tool_call.get("args", {})
            }
    return None

# Check if a tool needs approval
def is_sensitive_tool(tool_name):
    """Is this tool in our sensitive list?"""
    from src.agents.music_store_agent import sensitive_tool_names
    return tool_name in sensitive_tool_names

# Sidebar stuff
with st.sidebar:
    st.title("🎵 Music Store")
    st.markdown("---")
    
    if st.session_state.customer_id is None:
        # Not logged in yet
        st.write("### Customer Authentication")
        customers = get_customers()
        customer_options = [f"{name}" for id, name, count, spent in customers]
        customer_dict = {opt: cust[0] for opt, cust in zip(customer_options, customers)}
        selected_customer = st.selectbox(
            "Select a customer:",
            options=customer_options,
            placeholder="Choose a customer..."
        )
        if st.button("Start Session", type="primary", use_container_width=True):
            st.session_state.customer_id = str(customer_dict[selected_customer])
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
    else:
        # Already logged in
        st.info("🎵 Active Session")
        st.caption(f"Customer ID: {st.session_state.customer_id}")
        if st.button("End Session", type="secondary", use_container_width=True):
            # Reset everything when user logs out
            st.session_state.customer_id = None
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.session_state.pending_action = None
            st.rerun()

# Main chat UI
if st.session_state.customer_id:
    st.title("AI Music Assistant")
    st.markdown("---")
    
    # Show chat history
    messages_container = st.container()
    with messages_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
    
    # Config for the graph
    config = {
        "configurable": {
            "customer_id": int(st.session_state.customer_id),
            "thread_id": st.session_state.thread_id,
        }
    }
    
    # Check if we need to show the approval UI
    current_state = graph.get_state(config)
    
    
    # Look for pending tool calls that need approval
    has_pending_sensitive_tool = False
    tool_info = None
    
    # Method 1: Check next node
    if current_state and current_state.next == "sensitive_tools":
        has_pending_sensitive_tool = True
        state_values = current_state.values
        tool_info = extract_tool_call_info(state_values)
    
    # Method 2: Directly check messages (backup method)
    if not has_pending_sensitive_tool and current_state and hasattr(current_state, "values"):
        state_values = current_state.values
        for msg in reversed(state_values.get("messages", [])):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.get("name")
                if tool_name and is_sensitive_tool(tool_name):
                    # Check if there's already a response for this tool
                    tool_id = tool_call.get("id")
                    has_response = False
                    if tool_id:
                        for response_msg in state_values.get("messages", []):
                            if (hasattr(response_msg, "tool_call_id") and 
                                response_msg.tool_call_id == tool_id):
                                has_response = True
                                break
                    
                    if not has_response:
                        has_pending_sensitive_tool = True
                        tool_info = {
                            "name": tool_name,
                            "args": tool_call.get("args", {})
                        }
                        break
    
    # Show the approval UI if needed
    if has_pending_sensitive_tool and st.session_state.pending_action is None:
        # Add separator
        st.markdown("---")
        
        # Create container for approval UI
        with st.container():
            # Blue header with lock icon
            st.markdown("""
            <div style="padding: 15px; background-color: #0e4984; border-radius: 5px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: white;">🔐 Approval Required: Sensitive Operation</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if tool_info:
                # Layout for showing operation info
                col_info1, col_info2 = st.columns([1, 2])
                
                with col_info1:
                    st.markdown("<div style='font-weight: bold;'>Operation Type:</div>", unsafe_allow_html=True)
                    st.markdown("<div style='font-weight: bold;'>Description:</div>", unsafe_allow_html=True)
                
                with col_info2:
                    # Pretty-format the name
                    formatted_name = tool_info["name"].replace("_", " ").title()
                    st.markdown(f"<div>{formatted_name}</div>", unsafe_allow_html=True)
                    
                    # Show what this operation does
                    if tool_info["name"] == "update_profile":
                        st.markdown("<div>This operation will update your account information.</div>", unsafe_allow_html=True)
                    elif tool_info["name"] == "process_purchase":
                        st.markdown("<div>This operation will finalize your music purchase.</div>", unsafe_allow_html=True)
                    elif tool_info["name"] == "request_refund":
                        st.markdown("<div>This operation will process a refund request.</div>", unsafe_allow_html=True)
                
                # Divider
                st.markdown("""<hr style="margin: 15px 0;">""", unsafe_allow_html=True)
                
                # Details section header
                st.markdown("<h4>Operation Details</h4>", unsafe_allow_html=True)
                
                # Show the operation details in a nice format
                if tool_info["name"] == "update_profile" and "updates" in tool_info["args"]:
                    # Profile updates - show each field being changed
                    updates = tool_info["args"]["updates"]
                    
                    for field, value in updates.items():
                        field_label = field.replace("_", " ").title()
                        st.markdown(f"**{field_label}:** {value}")
                elif tool_info["name"] == "process_purchase" and "track_ids" in tool_info["args"]:
                    # Purchase - show summary and details
                    st.markdown(f"**Purchase Summary:** {len(tool_info['args']['track_ids'])} tracks")
                    with st.expander("View purchase details"):
                        st.json(tool_info["args"])
                else:
                    # Other operations - just show the JSON
                    st.json(tool_info["args"])
                
                # Another divider before buttons
                st.markdown("""<hr style="margin: 15px 0;">""", unsafe_allow_html=True)
                
                # Buttons row
                button_cols = st.columns([1, 1, 2])
                
                with button_cols[0]:
                    approve_btn = st.button("✅ Approve", type="primary", use_container_width=True)
                    if approve_btn:
                        st.session_state.pending_action = "approve"
                        st.rerun()
                
                with button_cols[1]:
                    cancel_btn = st.button("❌ Cancel", type="secondary", use_container_width=True)
                    if cancel_btn:
                        st.session_state.pending_action = "cancel"
                        st.rerun()
                
                with button_cols[2]:
                    st.markdown("""
                    <div style="padding-left: 10px; padding-top: 8px; font-size: 0.85em; color: #888;">
                    Please review the operation details before proceeding
                    </div>
                    """, unsafe_allow_html=True)
            
            else:
                # Fallback when we don't have details
                st.warning("⚠️ Sensitive operation detected, but details are not available")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", type="primary", use_container_width=True):
                        st.session_state.pending_action = "approve"
                        st.rerun()
                with col2:
                    if st.button("❌ Cancel", type="secondary", use_container_width=True):
                        st.session_state.pending_action = "cancel"
                        st.rerun()
        
        # Final separator
        st.markdown("---")
    
    # Handle the user's decision for pending actions
    if st.session_state.pending_action:
        try:
            with st.spinner("Processing your decision..."):
                if st.session_state.pending_action == "approve":
                    # User approved - just continue
                    result = graph.invoke(Command(resume=True), config=config)
                elif st.session_state.pending_action == "cancel":
                    # User cancelled - need to create a proper response
                    state_values = current_state.values
                    tool_info = extract_tool_call_info(state_values)
                    
                    if tool_info:
                        # Find the message with the tool call
                        last_message = None
                        for msg in reversed(state_values["messages"]):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                last_message = msg
                                break
                        
                        if last_message:
                            # Add a response for each tool call
                            for tool_call in last_message.tool_calls:
                                tool_id = tool_call.get("id")
                                tool_name = tool_call.get("name")
                                if tool_id:
                                    # Create cancelled response
                                    error_response = {"status": "cancelled", "message": "User cancelled this operation"}
                                    # Add to messages list
                                    state_values["messages"].append({
                                        "role": "tool",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "content": str(error_response)
                                    })
                    
                    # Add a message from the user
                    cancel_message = "Operation was cancelled, what else can I help you with today?"
                    state_values["messages"].append(("user", cancel_message))
                    
                    # Update the state and continue
                    graph.update_state(config, state_values)
                    result = graph.invoke(None, config=config)
                
                # Show the assistant's response
                if result and "messages" in result:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, "content"):
                        assistant_response = last_msg.content
                    else:
                        assistant_response = str(last_msg)
                    
                    # Handle dict responses
                    if isinstance(assistant_response, dict):
                        assistant_response = assistant_response.get("message", str(assistant_response))
                    
                    # Add to chat history
                    st.chat_message("assistant").write(assistant_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                
                # Clear the pending action
                st.session_state.pending_action = None
        except Exception as e:
            st.error(f"Error processing your decision: {str(e)}")
            st.session_state.pending_action = None
    
    # Chat input box
    if prompt := st.chat_input("How can I help you with music today?"):
        # Add user message to chat
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            with st.spinner("Processing your request..."):
                # Check if the agent is paused
                current_state = graph.get_state(config)
                
                if current_state and current_state.next:
                    # Agent is paused - add message and continue
                    state_values = current_state.values
                    state_values["messages"] = state_values["messages"] + [("user", prompt)]
                    result = graph.invoke(state_values, config=config)
                else:
                    # New conversation
                    state = {
                        "messages": [("user", prompt)],
                        "customer_info": {},
                        "selected_tracks": None
                    }
                    result = graph.invoke(state, config=config)
                
                # Get the response
                if result and "messages" in result:
                    last_msg = result["messages"][-1]
                    if hasattr(last_msg, "content"):
                        assistant_response = last_msg.content
                    else:
                        assistant_response = str(last_msg)
                    
                    # Fix dict responses
                    if isinstance(assistant_response, dict):
                        assistant_response = assistant_response.get("message", str(assistant_response))
                
                # Important: Check for sensitive tools BEFORE showing response
                current_state = graph.get_state(config)
                
                # Check if we need approval
                has_sensitive_tool_call = False
                
                # Method 1: Check next node
                if current_state and current_state.next == "sensitive_tools":
                    has_sensitive_tool_call = True
                else:
                    # Method 2: Check messages directly
                    if result and "messages" in result:
                        last_messages = result["messages"]
                        for msg in reversed(last_messages):
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get("name")
                                    if tool_name and is_sensitive_tool(tool_name):
                                        has_sensitive_tool_call = True
                                        break
                            if has_sensitive_tool_call:
                                break
                
                # If sensitive, show message + approval UI
                if has_sensitive_tool_call:
                    st.chat_message("assistant").write(assistant_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    # Force rerun to show approval UI
                    st.rerun()
                else:
                    # No approval needed, just show response
                    st.chat_message("assistant").write(assistant_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                    
                # One final check in case we missed something
                current_state = graph.get_state(config)
                if current_state and current_state.next == "sensitive_tools":
                    st.rerun()
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.chat_message("assistant").write(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
else:
    # Welcome screen
    st.title("🎵 Welcome to AI Music Store Assistant")
    st.markdown("""
    This demo showcases an AI assistant that helps customers:
    - 🔍 Discover new music based on preferences
    - 💳 Manage purchases and refunds
    - 👤 Update their profile information
    - 📜 View purchase history

    **To begin, please select a customer from the sidebar.**
    """)
