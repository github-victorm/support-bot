# Music Store Support Bot Demo

A customer support chatbot for a music store, built using LangGraph, LangSmith, and LangChain. This demo showcases advanced features like human-in-the-loop approvals, persistent conversation state, and secure customer data handling.

## 🎯 Key Features

- **Intelligent Music Recommendations**: Semantic search over music catalog using embeddings
- **Secure Customer Data Access**: Role-based access control for customer information
- **Human-in-the-Loop Purchase Approvals**: Staff review and approval workflow
- **Persistent Conversation State**: SQLite-based thread management
- **LangSmith Integration**: Full observability and tracing
- **Modern Streamlit UI**: Beautiful, responsive interface

## 🏗️ Architecture

### Core Components

1. **Vector Store (Chroma)**

   - Stores embeddings for the entire music catalog
   - Enables semantic search and recommendations
   - Persists to disk for fast startup
   - Uses OpenAI embeddings for high-quality results

2. **LangGraph Workflow**

   ```
   START
     │
     ▼
   Assistant ◄─────┐
     │            │
     ▼            │
   Tools ─────────┘
     │
     ▼
   Human Approval
     │
     ▼
    END
   ```

   - Cyclic graph for complex conversations
   - Tool executor for database operations
   - Human-in-the-loop node for purchase approvals
   - State persistence using SQLite

3. **State Management**

   - Thread-based conversation tracking
   - SQLite persistence for chat history
   - Customer session management
   - Secure state isolation between users

4. **Security Model**
   - Customer authentication required
   - Role-based access control
   - Data isolation between customers
   - Staff approval requirements

## 💡 Technical Decisions

### Why LangGraph?

1. **Complex Workflows**: The support bot needs to handle multiple types of queries (music recommendations, purchases) with different requirements. LangGraph's graph-based approach allows us to model these workflows explicitly.

2. **State Management**: LangGraph's state management capabilities are crucial for maintaining conversation context and handling multi-step processes like purchase approvals.

3. **Human-in-the-Loop**: Built-in support for human intervention points, essential for purchase approvals and moderation.

### Why SQLite for Persistence?

1. **Simplicity**: No need for a separate database server, perfect for demos and small to medium deployments.
2. **Reliability**: ACID compliance ensures conversation state is never lost.
3. **Performance**: Fast reads and writes for chat history and state management.

### Why Chroma for Vector Store?

 1. **Similarity Search**: I know from previous experience generating job recommendations for users that SQL alone can be a bit limiting. Using a vector db allows us to generate better recommendations with little effort.

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

## 📊 LangSmith Integration

The bot is fully integrated with LangSmith for:

- Conversation tracing
- Tool execution monitoring
- Performance analytics
- Error tracking
- A/B testing capabilities

To enable tracing:

1. Set `LANGCHAIN_API_KEY` in `.env`
2. Set `LANGCHAIN_PROJECT` to your project name
3. Set `LANGCHAIN_TRACING_V2=true`

## 🔒 Security Considerations

1. **Customer Data Protection**

   - Customers can only access their own data
   - Thread isolation prevents data leakage
   - API keys stored securely in environment variables

2. **Purchase Security**
   - All purchases require staff approval
   - Secure transaction handling
   - Audit trail in LangSmith

## 🎨 UI/UX Features

1. **Responsive Design**

   - Clean, modern interface
   - Real-time purchase approval status
   - Clear visual feedback

2. **Session Management**
   - Persistent chat history
   - Thread ID display
   - Login/logout functionality

## 📈 Future Enhancements

1. **Scalability**

   - Redis for state management in production
   - Load balancing for multiple instances
   - Caching layer for frequent queries

2. **Features**
   - Playlist recommendations
   - Bundle pricing
   - Customer reviews integration
   - Advanced analytics
