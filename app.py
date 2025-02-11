import streamlit as st
import sqlite3
from music_store_agent import MusicStoreSupport
from typing import List, Tuple
import pandas as pd
import os

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "support_bot" not in st.session_state:
    st.session_state.support_bot = MusicStoreSupport()
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def get_customers() -> List[Tuple[int, str]]:
    """Get list of customers from database for demo purposes"""
    conn = sqlite3.connect('chinook.db')
    cursor = conn.cursor()
    cursor.execute("SELECT CustomerId, FirstName || ' ' || LastName as Name FROM Customer")
    customers = cursor.fetchall()
    conn.close()
    return customers

def get_chat_history(thread_id: str) -> List[dict]:
    """Get chat history for a thread from SQLite"""
    try:
        db_path = "state_db/chat_memory.db"
        if not os.path.exists(db_path):
            return []
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM state WHERE config->>'thread_id' = ? ORDER BY created_at ASC",
            (thread_id,)
        )
        states = cursor.fetchall()
        conn.close()
        
        # Extract messages from states
        messages = []
        for state in states:
            if state[0] and 'messages' in state[0]:
                messages.extend(state[0]['messages'])
        return messages
    except Exception as e:
        st.error(f"Error loading chat history: {str(e)}")
        return []

# Page config
st.set_page_config(
    page_title="Music Store Support",
    page_icon="🎵",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f0f0f0;
        margin-right: 2rem;
    }
    .recommendation-feedback {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .purchase-status {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .state-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎵 Music Store Support")

# Authentication sidebar
with st.sidebar:
    st.header("Customer Login")
    if not st.session_state.authenticated:
        customers = get_customers()
        customer_dict = {f"{id}: {name}": id for id, name in customers}
        selected_customer = st.selectbox(
            "Select a customer (Demo Mode)",
            options=list(customer_dict.keys())
        )
        
        if st.button("Login"):
            customer_id = customer_dict[selected_customer]
            st.session_state.support_bot.authenticate_customer(customer_id)
            st.session_state.authenticated = True
            
            # Load existing chat history for this customer's thread
            thread_id = st.session_state.support_bot.get_thread_id(customer_id)
            if thread_id:
                chat_history = get_chat_history(thread_id)
                st.session_state.messages = chat_history
            
            st.rerun()
    else:
        st.success("Authenticated! ✅")
        
        # Show thread ID and current state
        thread_id = st.session_state.support_bot.get_thread_id(
            st.session_state.support_bot._customer_id
        )
        st.info(f"Session ID: {thread_id}")
        
        # Get current state for status display
        if thread_id:
            try:
                state = st.session_state.support_bot.graph.get_state(
                    {"configurable": {"thread_id": thread_id}}
                )
                if state:
                    st.markdown("### Current State")
                    if state.get("recommendation_context"):
                        st.markdown(
                            f"""<div class="state-info">
                            🎵 In recommendation cycle (iteration {state['recommendation_context'].get('iteration', 0)})
                            </div>""",
                            unsafe_allow_html=True
                        )
                    if state.get("interrupt"):
                        st.markdown(
                            '<div class="state-info">⏳ Waiting for purchase approval</div>',
                            unsafe_allow_html=True
                        )
            except Exception as e:
                st.error(f"Error getting state: {str(e)}")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.rerun()
    
    # Staff Approval Section
    st.markdown("---")
    st.header("Staff Actions")
    
    if thread_id:
        try:
            state = st.session_state.support_bot.graph.get_state(
                {"configurable": {"thread_id": thread_id}}
            )
            
            if state and state.get("interrupt"):
                interrupt_data = state["interrupt"]
                with st.container():
                    st.warning(f"⚠️ {interrupt_data.get('title', 'Approval Required')}")
                    st.info(interrupt_data.get("value", ""))
                    st.markdown(f"_{interrupt_data.get('description', '')}_")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Approve"):
                            response = st.session_state.support_bot.continue_from_interrupt(thread_id, True)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.rerun()
                    with col2:
                        if st.button("❌ Reject"):
                            response = st.session_state.support_bot.continue_from_interrupt(thread_id, False)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.rerun()
            else:
                st.info("No pending approvals")
        except Exception as e:
            st.error(f"Error checking interrupts: {str(e)}")
    else:
        st.info("No active session")

# Main chat interface
if st.session_state.authenticated:
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                st.markdown(f'<div class="chat-message user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f'<div class="chat-message bot-message">Support: {message["content"]}</div>', unsafe_allow_html=True)
                
                # Show feedback options if in recommendation cycle
                if thread_id:
                    state = st.session_state.support_bot.graph.get_state(
                        {"configurable": {"thread_id": thread_id}}
                    )
                    if state and state.get("recommendation_context"):
                        with st.container():
                            st.markdown(
                                '<div class="recommendation-feedback">How would you like to refine these recommendations?</div>',
                                unsafe_allow_html=True
                            )
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                if st.button("Similar Tracks"):
                                    st.session_state.messages.append({"role": "user", "content": "Show me more similar tracks"})
                                    st.rerun()
                            with col2:
                                if st.button("Different Genre"):
                                    st.session_state.messages.append({"role": "user", "content": "Show me different genres"})
                                    st.rerun()
                            with col3:
                                if st.button("Different Artist"):
                                    st.session_state.messages.append({"role": "user", "content": "Show me different artists"})
                                    st.rerun()
                            with col4:
                                if st.button("Purchase"):
                                    st.session_state.messages.append({"role": "user", "content": "I'd like to purchase these tracks"})
                                    st.rerun()

    # Chat input
    with st.container():
        # Check if we're in an interrupted state
        state = st.session_state.support_bot.graph.get_state(
            {"configurable": {"thread_id": thread_id}}
        ) if thread_id else None
        
        if state and state.get("interrupt"):
            st.info("⏳ Waiting for staff approval...")
            input_placeholder = "Please wait while your request is being reviewed..."
            user_input = st.text_input("Your message:", key="user_input", 
                                     placeholder=input_placeholder, disabled=True)
        else:
            input_placeholder = "How can I help you today? (e.g., 'Can you recommend some rock music?' or 'I want to purchase some tracks')"
            user_input = st.text_input("Your message:", key="user_input", placeholder=input_placeholder)
            
            if st.button("Send"):
                if user_input:
                    # Add user message to chat
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Get bot response
                    response = st.session_state.support_bot.chat(user_input)
                    
                    # Add bot response to chat
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Clear input and rerun
                    st.rerun()
else:
    st.info("👈 Please login using the sidebar to start chatting!")

# Footer
st.markdown("---")
st.markdown("Made using LangGraph and Streamlit") 