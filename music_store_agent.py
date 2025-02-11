from typing import Annotated, Dict, List, Optional, TypedDict, Literal
from datetime import datetime
import os
from operator import itemgetter
import uuid
import sqlite3
from sqlalchemy import create_engine, text

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from dotenv import load_dotenv
from langsmith import Client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize LangSmith for tracking
if os.getenv("LANGSMITH_API_KEY"):
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "music-store-support"
        # Add environment tag for better filtering
        os.environ["LANGCHAIN_TAGS"] = "production" if os.getenv("PRODUCTION") else "development"
        # Reduce tracing latency in non-serverless environments
        os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"
        client = Client()
        print("LangSmith tracking enabled")
    except Exception as e:
        print(f"Warning: Failed to initialize LangSmith: {str(e)}")
        client = None
else:
    print("Warning: LANGSMITH_API_KEY not found. Tracing disabled.")
    client = None

# Initialize SQLAlchemy engine
engine = create_engine('sqlite:///chinook.db')

def get_db():
    """Get SQLDatabase instance for LangChain tools"""
    return SQLDatabase.from_uri("sqlite:///chinook.db")

def query_db(query: str, parameters: Optional[Dict] = None) -> List[Dict]:
    """Execute a query using SQLAlchemy and return results as dictionaries"""
    with engine.connect() as conn:
        if parameters:
            result = conn.execute(text(query), parameters)
        else:
            result = conn.execute(text(query))
        # Convert to list of dictionaries
        return [dict(row._mapping) for row in result]

# Initialize embeddings and vector store with error handling
try:
    embeddings = OpenAIEmbeddings()
    vector_store = None  # Will be initialized in setup_vector_store
except Exception as e:
    raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

def setup_vector_store():
    """Initialize vector store with music catalog"""
    try:
        # First try to load existing vector store so we don't have to generate embeddings again
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./music_store_vectors"
        )
        
        # Check if the store is empty
        if vector_store._collection.count() > 0:
            print("Using existing vector store")
            return vector_store
        
        print("Vector store empty, generating embeddings...")
        
        # Get all tracks with metadata using SQLAlchemy
        tracks = query_db("""
            SELECT 
                t.TrackId,
                t.Name as Track,
                ar.Name as Artist,
                al.Title as Album,
                g.Name as Genre,
                t.UnitPrice as Price,
                t.Composer,
                al.AlbumId,
                ar.ArtistId,
                g.GenreId
            FROM Track t
            JOIN Album al ON t.AlbumId = al.AlbumId
            JOIN Artist ar ON al.ArtistId = ar.ArtistId
            JOIN Genre g ON t.GenreId = g.GenreId
        """)
        
        # Convert to documents for vector store
        documents = []
        for track in tracks:
            # Create rich text representation for embedding
            content = f"{track['Track']} by {track['Artist']} from album {track['Album']} in genre {track['Genre']}"
            if track.get('Composer'):
                content += f" composed by {track['Composer']}"
            
            # Store all metadata for retrieval
            metadata = {
                'track_id': track['TrackId'],
                'track_name': track['Track'],
                'artist': track['Artist'],
                'album': track['Album'],
                'genre': track['Genre'],
                'price': track['Price'],
                'composer': track.get('Composer'),
                'album_id': track['AlbumId'],
                'artist_id': track['ArtistId'],
                'genre_id': track['GenreId']
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create and persist vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="./music_store_vectors"
        )
        vector_store.persist()
        print(f"Generated embeddings for {len(documents)} tracks")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vector store: {str(e)}")

# Initialize vector store
vector_store = setup_vector_store()

# Define valid states
VALID_STATES = Literal["assistant", "recommend", "purchase", "human_approval", "END"]

# Define state types
class AgentState(TypedDict):
    """State for the music store support system"""
    messages: List[Dict]
    customer_id: Optional[int]
    next: VALID_STATES
    recommendation_feedback: Optional[str] = None  # Store user feedback about recommendations
    recommendation_context: Optional[Dict] = None  # Store context about previous recommendations
    pending_purchase: Optional[Dict] = None  # Store track/playlist info and total price
    human_approved: Optional[bool] = None

# Core tools
@tool
def semantic_music_search(query: str) -> str:
    """Search for music using natural language understanding"""
    docs = vector_store.similarity_search(
        query,
        k=5,
        search_type="mmr"  # Maximum Marginal Relevance for diversity
    )
    
    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "track": meta["track_name"],
            "artist": meta["artist"],
            "album": meta["album"],
            "genre": meta["genre"],
            "price": meta["price"],
            "track_id": meta["track_id"]  # Added to support purchase creation
        })
    return str(results)

@tool
def get_recommendations(customer_id: int, feedback: Optional[str] = None, previous_tracks: Optional[List[Dict]] = None) -> str:
    """Get personalized music recommendations based on customer's purchase history and feedback"""
    if MusicStoreSupport._current_customer != customer_id:
        return "Access denied. You can only view your own recommendations."
    
    db = get_db()
    
    # Build query based on feedback and previous recommendations
    if feedback and previous_tracks:
        # Extract genres, artists, and other features from previous recommendations
        previous_genres = {track['genre'] for track in previous_tracks}
        previous_artists = {track['artist'] for track in previous_tracks}
        
        # Analyze feedback to adjust recommendations
        feedback_lower = feedback.lower()
        if "different genre" in feedback_lower:
            # Exclude previously recommended genres
            docs = vector_store.similarity_search(
                "music NOT in genres " + " or ".join(previous_genres),
                k=5,
                search_type="mmr"
            )
        elif "similar genre" in feedback_lower:
            # Focus on the same genres but different artists
            genre_queries = [f"music in genre {genre}" for genre in previous_genres]
            docs = []
            for query in genre_queries:
                docs.extend(vector_store.similarity_search(
                    query,
                    k=2,
                    search_type="mmr",
                    filter={"artist": {"$nin": list(previous_artists)}}
                ))
        elif "different artist" in feedback_lower:
            # Same genres, different artists
            docs = vector_store.similarity_search(
                "music in genres " + " or ".join(previous_genres),
                k=5,
                search_type="mmr",
                filter={"artist": {"$nin": list(previous_artists)}}
            )
        else:
            # Use feedback as additional context for semantic search
            docs = vector_store.similarity_search(
                f"music similar to previous recommendations but {feedback}",
                k=5,
                search_type="mmr"
            )
    else:
        # Initial recommendations based on purchase history
        genres_query = """
        WITH CustomerGenres AS (
            SELECT g.Name as Genre, COUNT(*) as Frequency
            FROM Invoice i
            JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
            JOIN Track t ON il.TrackId = t.TrackId
            JOIN Genre g ON t.GenreId = g.GenreId
            WHERE i.CustomerId = ?
            GROUP BY g.GenreId
            ORDER BY Frequency DESC
            LIMIT 3
        )
        SELECT Genre FROM CustomerGenres
        """
        favorite_genres = db.run(genres_query, parameters=(customer_id,))
        
        docs = []
        for genre in favorite_genres:
            genre_query = f"music in genre {genre['Genre']}"
            docs.extend(vector_store.similarity_search(
                genre_query,
                k=2,
                search_type="mmr",
                filter={"genre": genre['Genre']}
            ))
    
    # Convert results to standard format
    results = []
    for doc in docs:
        meta = doc.metadata
        results.append({
            "track": meta["track_name"],
            "artist": meta["artist"],
            "album": meta["album"],
            "genre": meta["genre"],
            "price": meta["price"],
            "track_id": meta["track_id"]
        })
    
    return str(results)

@tool
def get_customer_info(customer_id: int) -> str:
    """Get customer information and order history"""
    # Skip access check for initial authentication
    if MusicStoreSupport._current_customer is not None and MusicStoreSupport._current_customer != customer_id:
        return "Access denied. You can only view your own information."
    
    db = get_db()
    return db.run("""
        SELECT 
            c.*,
            COUNT(DISTINCT i.InvoiceId) as OrderCount,
            SUM(i.Total) as TotalSpent
        FROM Customer c
        LEFT JOIN Invoice i ON c.CustomerId = i.CustomerId
        WHERE c.CustomerId = ?
        GROUP BY c.CustomerId
    """, parameters=(customer_id,))

@tool
def calculate_purchase_total(tracks: List[Dict]) -> str:
    """Calculate total price for a list of tracks"""
    total = sum(float(track['price']) for track in tracks)
    return f"${total:.2f}"

@tool
def process_purchase(customer_id: int, tracks: List[Dict]) -> str:
    """Process a purchase request for tracks"""
    if MusicStoreSupport._current_customer != customer_id:
        return "Access denied. You can only make purchases for your own account."
    
    total = sum(float(track['price']) for track in tracks)
    
    # Return the tracks and total for approval
    return f"Purchase request: ${total:.2f} for tracks: {', '.join(t['track'] for t in tracks)}"

def create_invoice(customer_id: int, tracks: List[Dict]) -> str:
    """Create a new invoice and invoice lines in the database"""
    total = sum(float(track['price']) for track in tracks)
    current_time = datetime.now().isoformat()
    
    try:
        # Create new invoice using SQLAlchemy
        result = query_db(
            """
            INSERT INTO Invoice (CustomerId, InvoiceDate, Total)
            VALUES (:customer_id, :invoice_date, :total)
            RETURNING InvoiceId
            """,
            {
                "customer_id": customer_id,
                "invoice_date": current_time,
                "total": total
            }
        )
        
        invoice_id = result[0]['InvoiceId']
        
        # Add invoice lines
        for track in tracks:
            query_db(
                """
                INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity)
                VALUES (:invoice_id, :track_id, :price, 1)
                """,
                {
                    "invoice_id": invoice_id,
                    "track_id": track['track_id'],
                    "price": track['price']
                }
            )
        
        return f"Purchase successful! Invoice #{invoice_id} created for ${total:.2f}. Tracks: {', '.join(t['track'] for t in tracks)}"
    except Exception as e:
        return f"Purchase failed: {str(e)}"

def recommendation_node(state: AgentState) -> AgentState:
    """Handle music recommendations and feedback cycles"""
    tools = [semantic_music_search, get_recommendations]
    tool_executor = ToolExecutor(tools)
    
    # Get previous recommendations and feedback from state
    feedback = state.get("recommendation_feedback")
    context = state.get("recommendation_context", {})
    previous_tracks = context.get("previous_tracks", [])
    
    # Execute the appropriate tool based on context
    if feedback:
        result = tool_executor.invoke({
            "name": "get_recommendations",
            "arguments": {
                "customer_id": state["customer_id"],
                "feedback": feedback,
                "previous_tracks": previous_tracks
            }
        })
    else:
        result = tool_executor.invoke({
            "name": "get_recommendations",
            "arguments": {
                "customer_id": state["customer_id"]
            }
        })
    
    # Update recommendation context
    new_tracks = eval(result)  # Convert string result back to list
    context["previous_tracks"] = previous_tracks + new_tracks
    context["iteration"] = context.get("iteration", 0) + 1
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=str(result))],
        "recommendation_context": context,
        "next": "assistant"  # Return to assistant for feedback or next step
    }

def purchase_node(state: AgentState) -> AgentState:
    """Handle purchase calculations and processing"""
    tools = [calculate_purchase_total, process_purchase]
    tool_executor = ToolExecutor(tools)
    
    # Extract the last message to determine what to purchase
    last_message = state["messages"][-1].content
    
    # First calculate total
    if "calculate" in last_message.lower():
        tracks = eval(last_message.split("tracks=")[-1])  # Extract tracks from message
        result = tool_executor.invoke({
            "name": "calculate_purchase_total",
            "arguments": {"tracks": tracks}
        })
    else:
        # Process purchase
        tracks = eval(last_message.split("tracks=")[-1])
        result = tool_executor.invoke({
            "name": "process_purchase",
            "arguments": {
                "customer_id": state["customer_id"],
                "tracks": tracks
            }
        })
        # Store tracks for later invoice creation
        state["pending_purchase"] = {"tracks": tracks}
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=str(result))],
        "next": "assistant"
    }

def assistant(state: AgentState) -> AgentState:
    """Primary assistant that handles all queries"""
    messages = state["messages"]
    customer_id = state["customer_id"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert music store assistant that helps customers discover and purchase music.
        You can:
        1. Recommend music based on their tastes and history using semantic search
        2. Search for specific songs, artists, or albums using natural language
        3. Process purchases (all purchases require staff approval for security)
        4. View customer information and purchase history
        
        Available tools:
        - semantic_music_search: Search music catalog using natural language
        - get_recommendations: Get personalized recommendations based on purchase history and feedback
        - get_customer_info: Look up customer information
        - calculate_purchase_total: Calculate total price for tracks
        - process_purchase: Handle purchase requests (requires staff approval)
        
        Always be enthusiastic about music and professional with customer service!
        
        When making recommendations:
        1. Route to 'recommend' node with initial request
        2. After recommendations, ask if they want:
           - More similar tracks
           - Different genres
           - Different artists
           - Or to proceed with purchase
        3. Route to 'recommend' again with their feedback
        4. Continue until they're satisfied
        
        When handling purchases:
        1. Route to 'purchase' node with tracks to calculate total
        2. If they confirm, route to 'purchase' node to process
        3. Inform them that staff approval is required
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    response = ChatOpenAI(model="gpt-4-mini").invoke(
        prompt.format_messages(messages=messages),
        config={
            "tags": ["assistant"],
            "metadata": {
                "customer_id": customer_id,
                "timestamp": datetime.now().isoformat()
            }
        }
    )
    
    # Check if this is a purchase request that needs approval
    if "process_purchase" in response.content.lower():
        from langgraph.prebuilt import interrupt
        return interrupt(
            value=response.content,
            key="purchase_approval",
            title="Purchase Approval Required",
            description="Please review and approve or reject this purchase",
            inputs={
                "approved": {
                    "type": "boolean",
                    "title": "Approve Purchase",
                    "description": "Click true to approve, false to reject"
                },
                "tracks": {
                    "type": "array",
                    "title": "Tracks to Purchase",
                    "description": "The tracks being purchased"
                }
            }
        )
    
    # Store feedback if we're in a recommendation cycle
    if state.get("recommendation_context") and "feedback" in response.content.lower():
        state["recommendation_feedback"] = response.content
    
    # Determine next node based on content
    next_node = "END"
    if any(word in response.content.lower() for word in ["recommend", "similar", "different genre", "different artist"]):
        next_node = "recommend"
    elif any(word in response.content.lower() for word in ["purchase", "buy", "calculate"]):
        next_node = "purchase"
    
    return {
        **state,
        "messages": messages + [AIMessage(content=response.content)],
        "next": next_node
    }

def create_graph():
    """Create the agent graph with nodes and edges"""
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Initialize SQLite checkpointer for thread persistence
    db_path = "state_db/chat_memory.db"
    os.makedirs("state_db", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # Add nodes
    workflow.add_node("assistant", assistant)
    workflow.add_node("recommend", recommendation_node)
    workflow.add_node("purchase", purchase_node)
    
    # Add edges
    workflow.add_edge(START, "assistant")
    workflow.add_edge("recommend", "assistant")  # Recommendation cycle
    workflow.add_edge("purchase", "assistant")   # Purchase cycle
    
    # Add conditional edges from assistant
    workflow.add_conditional_edges(
        "assistant",
        lambda x: x["next"],
        {
            "recommend": "recommend",  # For recommendations and feedback
            "purchase": "purchase",    # For purchase processing
            END: END
        }
    )
    
    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)

class MusicStoreSupport:
    """Main interface for the music store support system"""
    _current_customer = None
    _thread_ids = {}
    _db_conn = None
    
    def __init__(self):
        try:
            self.graph = create_graph()
            self._customer_id = None
            global vector_store
            if vector_store is None:
                vector_store = setup_vector_store()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MusicStoreSupport: {str(e)}")
    
    def chat(self, message: str) -> str:
        """Process a customer message"""
        if not self._customer_id:
            return "Please authenticate first using authenticate_customer(customer_id)"

        # Create initial state
        state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": self._customer_id,
            "next": "assistant"
        }
        
        # Get thread ID for this customer session
        thread_id = self.get_thread_id(self._customer_id)
        
        try:
            # Run the graph with interrupt handling
            result = self.graph.invoke(
                state,
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "user_id": str(self._customer_id)
                    }
                }
            )
            
            if isinstance(result, dict) and "interrupt" in result:
                return "Waiting for staff approval..."
            
            return result["messages"][-1].content
            
        except Exception as e:
            if "interrupt" in str(e):
                return "Waiting for staff approval..."
            raise e

    def continue_from_interrupt(self, thread_id: str, approved: bool) -> str:
        """Continue execution after an interrupt"""
        try:
            if approved:
                # Get the state to access the purchase details
                state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
                if state and state.get("interrupt"):
                    # Create the invoice
                    result = create_invoice(self._customer_id, state["interrupt"]["tracks"])
                    return result
            
            return "Purchase was not approved. Please contact support if you have any questions."
            
        except Exception as e:
            return f"Error processing purchase: {str(e)}"
            
    def get_thread_id(self, customer_id: int) -> str:
        """Get the thread ID for a customer session"""
        return self._thread_ids.get(customer_id)
    
    def authenticate_customer(self, customer_id: int) -> None:
        """Authenticate a customer"""
        customer_info = get_customer_info(customer_id)
        if not customer_info:
            raise ValueError(f"Customer {customer_id} not found")
            
        self._customer_id = customer_id
        if customer_id not in self._thread_ids:
            self._thread_ids[customer_id] = str(uuid.uuid4())
        print(f"Authenticated as {customer_info[0]['FirstName']} {customer_info[0]['LastName']}")
        print(f"Session thread ID: {self.get_thread_id(customer_id)}")

if __name__ == "__main__":
    # Example usage
    support = MusicStoreSupport()
    support.authenticate_customer(1)
    print(support.chat("Can you recommend some rock music?"))