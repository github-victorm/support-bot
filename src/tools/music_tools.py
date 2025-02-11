from langchain_core.tools import tool
import logging
from ..utils.database import get_db
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# Initialize database
db = get_db()

@tool
def query_invoice_history(customer_id: str) -> str:
    """Get the customer's invoice and payment history."""
    try:
        query = """
        SELECT 
            i.InvoiceDate,
            i.Total,
            t.Name as TrackName,
            ar.Name as ArtistName,
            g.Name as Genre
        FROM Invoice i
        JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        JOIN Track t ON il.TrackId = t.TrackId
        JOIN Album al ON t.AlbumId = al.AlbumId
        JOIN Artist ar ON al.ArtistId = ar.ArtistId
        JOIN Genre g ON t.GenreId = g.GenreId
        WHERE i.CustomerId = ?
        ORDER BY i.InvoiceDate DESC
        LIMIT 10
        """
        result = db.run(query, [customer_id])
        return str(result)
    except Exception as e:
        return f"Error fetching invoice history: {str(e)}"

@tool
def get_recommendations(genre: str, vector_store: Chroma) -> str:
    """Get music recommendations based on a genre using vector search."""
    try:
        if not genre:
            logger.warning("No genre provided")
            return "Error: No genre provided"
            
        # Create a search query that emphasizes the genre
        search_query = f"Find music tracks in the {genre} genre"
        
        # Get initial recommendations using vector search
        results = vector_store.similarity_search_with_score(
            search_query,
            k=10  # Get top 10 results
        )
        
        # Grade the relevance of results
        relevant_results = []
        for doc, score in results:
            # Lower score means more similar in Chroma
            if score < 0.3:  # Adjust this threshold based on your needs
                metadata = doc.metadata
                relevant_results.append({
                    'track': metadata['track_name'],
                    'artist': metadata['artist'],
                    'album': metadata['album'],
                    'genre': metadata['genre'],
                    'score': score
                })
        
        # If we don't have enough relevant results, try a broader search
        if len(relevant_results) < 3:
            logger.info("Falling back to broader search strategy")
            broader_query = f"music similar to {genre} style"
            broader_results = vector_store.similarity_search_with_score(
                broader_query,
                k=15  # Try more results
            )
            
            for doc, score in broader_results:
                if score < 0.4:  # More lenient threshold for broader search
                    metadata = doc.metadata
                    if not any(r['track'] == metadata['track_name'] for r in relevant_results):
                        relevant_results.append({
                            'track': metadata['track_name'],
                            'artist': metadata['artist'],
                            'album': metadata['album'],
                            'genre': metadata['genre'],
                            'score': score
                        })
        
        if not relevant_results:
            logger.warning(f"No recommendations found for genre: {genre}")
            return f"I couldn't find any highly relevant recommendations for {genre}. Would you like to try a different genre or describe the type of music you're looking for?"
            
        # Sort by relevance score
        relevant_results.sort(key=lambda x: x['score'])
        
        # Format the results nicely
        recommendations = [f"Here are some recommendations based on your interest in {genre}:"]
        for result in relevant_results[:10]:  # Limit to top 10
            recommendations.append(
                f"- {result['track']} by {result['artist']} "
                f"(Album: {result['album']}, Genre: {result['genre']})"
            )
        
        return "\n".join(recommendations)
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}", exc_info=True)
        return f"Error getting recommendations: {str(e)}"

@tool
def update_memory(update_type: str) -> str:
    """Update customer profile or recommendations memory"""
    return f"Memory update requested: {update_type}" 