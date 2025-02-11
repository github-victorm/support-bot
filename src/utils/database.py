import sqlite3
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import logging

logger = logging.getLogger(__name__)

def get_db():
    """Get SQLDatabase instance"""
    conn = sqlite3.connect("chinook.db")
    conn.row_factory = sqlite3.Row
    return SQLDatabase.from_uri(
        "sqlite:///chinook.db", 
        include_tables=['Track', 'Album', 'Artist', 'Genre']
    )

def setup_vector_store(embeddings: OpenAIEmbeddings):
    """Initialize vector store with music catalog"""
    try:
        # First try to load existing vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "music_store_vectors")
        )
        
        # Check if the store is empty
        if vector_store._collection.count() > 0:
            logger.info("Using existing vector store")
            return vector_store
        
        logger.info("Vector store empty, generating embeddings...")
        
        # Use direct sqlite3 connection for better control
        conn = sqlite3.connect("chinook.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all tracks with metadata
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
            # Create rich text representation for embedding
            content = f"{row['Track']} by {row['Artist']} from album {row['Album']} in genre {row['Genre']}"
            if row['Composer']:
                content += f" composed by {row['Composer']}"
            
            # Store all metadata for retrieval
            metadata = {
                'track_id': row['TrackId'],
                'track_name': row['Track'] or "",
                'artist': row['Artist'] or "",
                'album': row['Album'] or "",
                'genre': row['Genre'] or "",
                'price': float(row['Price']) if row['Price'] is not None else 0.0,
                'composer': row['Composer'] or "",
                'album_id': row['AlbumId'],
                'artist_id': row['ArtistId'],
                'genre_id': row['GenreId']
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        conn.close()
        
        # Create vector store with persistence
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "music_store_vectors")
        )
        logger.info(f"Generated embeddings for {len(documents)} tracks")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to initialize vector store: {str(e)}") 