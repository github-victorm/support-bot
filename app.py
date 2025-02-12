import streamlit as st
import sqlite3
from typing import List, Tuple
from src.agents.music_store_agent import graph
from src.config.settings import setup_langsmith, check_api_keys
from src.utils.database import initialize_vector_store
import uuid
import time

# Page config
st.set_page_config(
    page_title="Music Store Agent",
    page_icon="🎵",
    layout="wide"
)

# Check for required API keys on startup
check_api_keys()

# Initialize LangSmith
client = setup_langsmith()

# Initialize vector store
with st.spinner("Initializing vector store..."):
    initialize_vector_store()
    st.success("✅ Vector store ready")

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "track_info" not in st.session_state:
    st.session_state.track_info = None

def get_customers() -> List[Tuple[int, str]]:
    """Get list of customers from database"""
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

# Sidebar for customer selection
with st.sidebar:
    st.title("🎵 Music Store")
    st.markdown("---")
    
    if st.session_state.customer_id is None:
        st.write("### Customer Authentication")
        customers = get_customers()
        
        # Enhanced customer selection with purchase history
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
        st.info(f"🎵 Active Session")
        st.caption(f"Customer ID: {st.session_state.customer_id}")
        if st.button("End Session", type="secondary", use_container_width=True):
            st.session_state.customer_id = None
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.session_state.track_info = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Example Queries")
        examples = [
            "What music do you recommend for a workout?",
            "Show me my recent purchases",
            "I want to update my email address",
            "Can you recommend some jazz music?",
            "I need a refund for my last purchase"
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

# Main chat interface
if st.session_state.customer_id:
    st.title("AI Music Assistant")
    st.markdown("---")
    
    # Initialize chat message container
    messages_container = st.container()
    
    # Display chat messages
    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input at the bottom
    if prompt := st.chat_input("How can I help you with music today?"):
        # Add user message to chat
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        message = None  # Store message outside the spinner
        
        try:
            with st.spinner("Processing your request..."):
                # Convert customer_id to integer for state and tools
                customer_id = int(st.session_state.customer_id)
                
                # Prepare the state and config for the graph
                state = {
                    "messages": [("user", prompt)],
                    "customer_info": {},  # Keep empty since we use config
                    "selected_tracks": None,
                    "track_info": st.session_state.track_info
                }
                
                config = {
                    "configurable": {
                        "customer_id": customer_id,  # Move to configurable for tool access
                        "thread_id": st.session_state.thread_id,
                        "track_info": st.session_state.track_info
                    }
                }
                
                # Check if this is a response to a tool call
                snapshot = graph.get_state(config)
                if snapshot and snapshot.next and prompt.strip().lower() == 'y':
                    # Just continue with None for approval
                    result = graph.invoke(None, config)
                elif snapshot and snapshot.next:
                    # Just send the user's feedback as a new message
                    result = graph.invoke(state, config)
                else:
                    # Normal message flow
                    result = graph.invoke(state, config)
                
                # Process assistant response
                if result and "messages" in result:
                    # Get only the last message from the assistant
                    last_message = result["messages"][-1]
                    content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    
                    # Handle dictionary responses
                    if isinstance(content, dict):
                        # Store track info if present
                        if "track_info" in content:
                            st.session_state.track_info = content["track_info"]
                            config["configurable"]["track_info"] = content["track_info"]
                        # Get the human-readable message
                        if "message" in content and isinstance(content["message"], str):
                            message = content["message"]
                    
                    # Handle string messages if they're meaningful
                    elif content and isinstance(content, str) and content.strip():
                        # Skip raw data responses
                        if not any([
                            content.startswith("{") and content.endswith("}"),  # JSON objects
                            content.startswith("[") and content.endswith("]"),  # JSON arrays
                            content.startswith("Tool Calls:"),  # Tool call info
                            "call_" in content,  # Tool call IDs
                            "toolu_" in content,  # Tool IDs
                        ]) and not content.strip().startswith(("Name:", "Call ID:", "Args:")):
                            message = content
                    
                # Check if we need approval for a sensitive tool
                snapshot = graph.get_state(config)
                needs_approval = snapshot and snapshot.next
            
            # Display messages outside the spinner
            if message:
                st.chat_message("assistant").write(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message
                })
            
            if needs_approval:
                st.info("Type 'y' to proceed or provide feedback if you want to change anything.")
        
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            st.chat_message("assistant").write(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
else:
    # Welcome screen when no customer is selected
    st.title("🎵 Welcome to AI Music Store Assistant")
    st.markdown("""
    This demo showcases an AI assistant that helps customers:
    - 🔍 Discover new music based on preferences
    - 💳 Manage purchases and refunds
    - 👤 Update their profile information
    - 📜 View purchase history
    
    **To begin, please select a customer from the sidebar.**
    """)

# Remove all the separate approval handling code since the graph handles it 