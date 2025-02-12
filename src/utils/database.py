import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import logging
from typing import Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Singleton instances
_vector_store_instance: Optional[Chroma] = None
_chroma_client: Optional[chromadb.PersistentClient] = None

def initialize_vector_store() -> None:
    """Initialize the vector store on startup. Should be called once when the app starts."""
    global _vector_store_instance, _chroma_client
    try:
        logger.info("Initializing vector store...")
        embeddings = OpenAIEmbeddings()
        _vector_store_instance = setup_vector_store(embeddings)
        logger.info("Vector store initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def get_db():
    """Get SQLDatabase instance"""
    conn = sqlite3.connect("chinook.db")
    conn.row_factory = sqlite3.Row
    return SQLDatabase.from_uri(
        "sqlite:///chinook.db", 
        include_tables=['Track', 'Album', 'Artist', 'Genre']
    )

def get_vector_store() -> Chroma:
    """Get the singleton instance of the vector store"""
    global _vector_store_instance
    if _vector_store_instance is None:
        raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
    return _vector_store_instance

def setup_vector_store(embeddings: OpenAIEmbeddings) -> Chroma:
    """Initialize vector store with music catalog"""
    try:
        global _chroma_client
        
        # Get absolute path to the project root directory
        persist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "music_store_vectors")
        collection_name = "langchain"
        
        # Initialize the persistent client
        if _chroma_client is None:
            logger.info("Initializing Chroma client...")
            _chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Try to get existing collection
        try:
            collection = _chroma_client.get_collection(name=collection_name)
            count = collection.count()
            if count > 0:
                logger.info(f"Found existing collection with {count} embeddings")
                return Chroma(
                    client=_chroma_client,
                    collection_name=collection_name,
                    embedding_function=embeddings
                )
        except ValueError:
            logger.info("No existing collection found")
        except Exception as e:
            logger.warning(f"Error accessing collection: {str(e)}")
        
        # If we get here, we need to create a new collection
        logger.info("Creating new collection...")
        
        # Get track data from database
        conn = sqlite3.connect("chinook.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
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
            for row in cursor.fetchall():
                content = f"{row['Track']} by {row['Artist']} from album {row['Album']} in genre {row['Genre']}"
                if row['Composer']:
                    content += f" composed by {row['Composer']}"
                
                metadata = {
                    'track_id': str(row['TrackId']),  # Chroma requires string IDs
                    'track_name': row['Track'] or "",
                    'artist': row['Artist'] or "",
                    'album': row['Album'] or "",
                    'genre': row['Genre'] or "",
                    'price': str(float(row['Price'])) if row['Price'] is not None else "0.0",  # Convert to string
                    'composer': row['Composer'] or "",
                    'album_id': str(row['AlbumId']),
                    'artist_id': str(row['ArtistId']),
                    'genre_id': str(row['GenreId'])
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
        finally:
            cursor.close()
            conn.close()
        
        if not documents:
            raise RuntimeError("No documents found in the music catalog")
        
        # Create new vector store using the client
        logger.info(f"Creating new vector store with {len(documents)} documents...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            client=_chroma_client  # Use the persistent client
        )
        
        logger.info("Vector store created successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to initialize vector store: {str(e)}") 