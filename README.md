# Music Store Support Bot Demo

A customer support chatbot for a digital music store, built using LangGraph, LangChain, and LangSmith. This demo showcases a conversational agent that helps customers discover and purchase music while managing their profile and purchase history.

## 🎯 Features

- **Music Discovery**

  - Semantic search for music recommendations based on:
    - Genre preferences
    - Mood/style descriptions
    - Artist similarity
    - Musical elements (instruments, tempo, etc.)

- **Purchase Management**

  - Browse and select tracks from recommendations
  - Preview purchase details before confirmation
  - Process secure transactions
  - View purchase history
  - Request refunds for previous purchases

- **Profile Management**
  - View customer information
  - Update contact details
  - Modify billing/shipping addresses
  - Update customer profile

## 🛠️ Tools

The agent uses several specialized tools to handle different tasks:

1. **Music Discovery**

   - `get_recommendations`: Semantic search for music based on any criteria
   - `parse_track_selection`: Process user's track selections from recommendations

2. **Purchase Management**

   - `process_purchase`: Handle music track purchases
   - `query_invoice_history`: Retrieve customer's purchase history
   - `request_refund`: Process refunds for previous purchases

3. **Profile Management**
   - `fetch_customer_info`: Get customer details
   - `update_profile`: Modify customer information

## 🚀 Getting Started

1. **Installation**

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Setup**

   ```bash
   cp .env.example .env
   # Add your OpenAI API key and LangSmith API key
   ```

3. **Run the App**
   ```bash
   streamlit run app.py
   ```

## 🔒 Security

- Human-in-the-loop approval for sensitive operations
- Secure customer data handling
- Transaction verification
- Customer authentication required

## 📊 LangSmith Integration

Enable tracing by setting in your `.env`:

```bash
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=your_project
LANGCHAIN_TRACING_V2=true
```
