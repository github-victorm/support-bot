from langchain_core.tools import tool
import logging
from ..utils.database import get_db, get_vector_store
from langchain_core.runnables import RunnableConfig
import sqlite3
import re

logger = logging.getLogger(__name__)

# Database path
DB_PATH = "chinook.db"

@tool
def get_recommendations(query: str, config: RunnableConfig) -> list[dict]:
    """Get music recommendations based on any search criteria (genre, mood, style, artist similarity, etc).
    Examples:
    - "Find upbeat rock songs"
    - "Songs similar to Hotel California"
    - "Jazz music with piano"
    - "Energetic workout music"
    - "Relaxing classical compositions"
    """
    try:
        if not query:
            return {
                "status": "error",
                "message": "Please provide a search query describing what kind of music you're looking for"
            }
        
        # Try to get the vector store instance
        try:
            vector_store = get_vector_store()
        except RuntimeError as e:
            # Fallback to database search if vector store is not available
            logger.warning(f"Vector store error: {str(e)}. Using database fallback.")
            return get_recommendations_fallback(query, config)
        
        # Create a simple retriever with MMR search for better result diversity
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )
        
        # Get recommendations using the newer invoke method
        docs = retriever.invoke(query)
        
        if not docs:
            return {
                "status": "error",
                "message": (
                    f"I couldn't find any relevant recommendations for '{query}'. "
                    "Try describing the type of music, mentioning specific artists, "
                    "genres, moods, or instruments."
                )
            }
        
        # Format results with track IDs for purchase
        tracks = []
        message_lines = [f"Here are some recommendations based on your search for '{query}':"]
        
        for i, doc in enumerate(docs, 1):
            track = {
                "track_id": int(doc.metadata['track_id']),
                "track_name": doc.metadata['track_name'],
                "artist": doc.metadata['artist'],
                "album": doc.metadata['album'],
                "genre": doc.metadata['genre'],
                "price": float(doc.metadata['price'])
            }
            tracks.append(track)
            
            message_lines.append(
                f"{i}. {track['track_name']} by {track['artist']} "
                f"(Album: {track['album']}, Genre: {track['genre']}, "
                f"Price: ${track['price']:.2f})"
            )
        
        # Return structured response directly as a dict
        return {
            "status": "success",
            "message": "\n".join(message_lines),
            "track_info": {str(i): track for i, track in enumerate(tracks, 1)},  # Map track numbers to track info
            "track_ids": [track["track_id"] for track in tracks]  # List of track IDs in order
        }
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {str(e)}")
        return {
            "status": "error",
            "message": f"Sorry, I encountered an error while getting recommendations: {str(e)}"
        }

@tool
def query_invoice_history(config: RunnableConfig) -> list[dict]:
    """Get the customer's invoice and payment history.
    returns a list of invoices.
    """
    customer_id = config.get("configurable", {}).get("customer_id")
    if not customer_id:
        raise ValueError("Customer ID is required")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # This will make rows accessible by column name
    cursor = conn.cursor()

    query = """
    SELECT 
        i.InvoiceId,
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

    cursor.execute(query, [customer_id])
    rows = cursor.fetchall()
    column_names = [column[0] for column in cursor.description]
    results = [dict(zip(column_names, row)) for row in rows]

    cursor.close()
    conn.close()

    return results

@tool
def fetch_customer_info(config: RunnableConfig) -> str:
    """Get customer information."""
    customer_id = config.get("configurable", {}).get("customer_id")
    if not customer_id:
        raise ValueError("No customer ID provided.")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT FirstName, LastName, Email, Phone, Address, City, State, Country, PostalCode, Company
            FROM Customer 
            WHERE CustomerId = ?
        """, (customer_id,))
        
        customer = cursor.fetchone()
        if not customer:
            return "Customer not found."
        
        return (
            f"Customer: {customer[0]} {customer[1]}\n"
            f"Email: {customer[2]}\n"
            f"Phone: {customer[3]}\n"
            f"Address: {customer[4]}, {customer[5]}, {customer[6]}, {customer[7]} {customer[8]}\n"
            f"Company: {customer[9]}"
        )
        
    except Exception as e:
        return f"Error retrieving customer information: {str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool
def request_refund(invoice_id: int, config: RunnableConfig) -> str:
    """Process a refund for the specified invoice."""
    customer_id = config.get("configurable", {}).get("customer_id")
    if not customer_id:
        raise ValueError("No customer ID provided.")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Verify invoice exists and belongs to customer
        cursor.execute("""
            SELECT Total FROM Invoice 
            WHERE InvoiceId = ? AND CustomerId = ?
        """, (invoice_id, customer_id))
        invoice = cursor.fetchone()
        if not invoice:
            return "Invoice not found or doesn't belong to this customer."
        
        # Check if already refunded
        cursor.execute("""
            SELECT 1 FROM Invoice 
            WHERE BillingAddress LIKE ? AND Total < 0
        """, (f"Refund for invoice {invoice_id}%",))
        if cursor.fetchone():
            return "This invoice has already been refunded."
        
        # Process refund
        cursor.execute("BEGIN TRANSACTION")
        
        # Create refund invoice
        cursor.execute("""
            INSERT INTO Invoice (CustomerId, InvoiceDate, BillingAddress, Total)
            VALUES (?, datetime('now'), ?, ?)
        """, (customer_id, f"Refund for invoice {invoice_id}", -float(invoice[0])))
        
        refund_id = cursor.lastrowid
        
        # Copy invoice lines with negative amounts
        cursor.execute("""
            INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity)
            SELECT ?, TrackId, -UnitPrice, Quantity
            FROM InvoiceLine WHERE InvoiceId = ?
        """, (refund_id, invoice_id))
        
        cursor.execute("COMMIT")
        return f"Refund processed successfully. Refund invoice #{refund_id} created for ${abs(float(invoice[0])):.2f}"
        
    except Exception as e:
        cursor.execute("ROLLBACK")
        return f"Error processing refund: {str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool
def process_purchase(track_ids: list[int], config: RunnableConfig) -> dict:
    """Process a music purchase transaction for the given tracks."""
    customer_id = config.get("configurable", {}).get("customer_id")
    if not customer_id:
        return {
            "status": "error",
            "message": "No customer ID provided."
        }
    
    if not track_ids:
        return {
            "status": "error",
            "message": "Please provide track IDs to purchase."
        }
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Verify customer exists
        cursor.execute(
            "SELECT CustomerId FROM Customer WHERE CustomerId = ?",
            (customer_id,)
        )
        if not cursor.fetchone():
            return {
                "status": "error",
                "message": "Customer not found."
            }
        
        # Get track details and calculate total
        track_details = []
        total_amount = 0
        
        for track_id in track_ids:
            cursor.execute("""
                SELECT t.Name, t.UnitPrice, ar.Name as ArtistName
                FROM Track t
                JOIN Album al ON t.AlbumId = al.AlbumId
                JOIN Artist ar ON al.ArtistId = ar.ArtistId
                WHERE t.TrackId = ?
            """, (track_id,))
            
            track = cursor.fetchone()
            if not track:
                return {
                    "status": "error",
                    "message": f"Track {track_id} not found."
                }
            
            price = float(track['UnitPrice'])
            total_amount += price
            track_details.append({
                'name': track['Name'],
                'artist': track['ArtistName'],
                'price': price
            })
        
        # Create invoice
        cursor.execute(
            "INSERT INTO Invoice (CustomerId, InvoiceDate, Total) VALUES (?, datetime('now'), ?)",
            (customer_id, total_amount)
        )
        invoice_id = cursor.lastrowid
        
        # Add invoice lines
        for track_id in track_ids:
            cursor.execute("""
                INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity)
                SELECT ?, TrackId, UnitPrice, 1
                FROM Track WHERE TrackId = ?
            """, (invoice_id, track_id))
        
        conn.commit()
        
        # Return purchase confirmation
        return {
            "status": "success",
            "message": "Purchase completed successfully!\n\n" + 
                      "Purchased tracks:\n" +
                      "\n".join([f"- {track['name']} by {track['artist']} (${track['price']:.2f})"
                                for track in track_details]) +
                      f"\n\nInvoice #{invoice_id}\nTotal: ${total_amount:.2f}",
            "invoice_id": invoice_id,
            "total": total_amount,
            "tracks": track_details
        }
        
    except Exception as e:
        logger.error(f"Error processing purchase: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing purchase: {str(e)}"
        }
    finally:
        cursor.close()
        conn.close()

@tool
def update_profile(updates: dict, config: RunnableConfig) -> str:
    """Update customer profile information.
    
    Args:
        updates: Dictionary with fields to update (first_name, last_name, company, address, 
                city, state, country, postal_code, phone, email)
    """
    customer_id = config.get("configurable", {}).get("customer_id")
    if not customer_id:
        raise ValueError("No customer ID provided.")
    
    if not updates:
        raise ValueError("No updates provided.")
    
    valid_fields = {
        'first_name': 'FirstName',
        'last_name': 'LastName',
        'company': 'Company',
        'address': 'Address',
        'city': 'City',
        'state': 'State',
        'country': 'Country',
        'postal_code': 'PostalCode',
        'phone': 'Phone',
        'email': 'Email'
    }
    
    # Validate fields
    invalid_fields = [f for f in updates.keys() if f not in valid_fields]
    if invalid_fields:
        return f"Invalid fields: {', '.join(invalid_fields)}"
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Verify customer exists
        cursor.execute("SELECT 1 FROM Customer WHERE CustomerId = ?", (customer_id,))
        if not cursor.fetchone():
            return "Customer not found."
        
        # Build update query dynamically based on provided fields
        set_clauses = [f"{valid_fields[field]} = ?" for field in updates.keys()]
        values = list(updates.values()) + [customer_id]
        
        cursor.execute(
            f"UPDATE Customer SET {', '.join(set_clauses)} WHERE CustomerId = ?",
            values
        )
        
        if cursor.rowcount == 0:
            return "No changes were made to the profile."
            
        conn.commit()
        return f"Profile updated successfully. Updated fields: {', '.join(updates.keys())}"
        
    except Exception as e:
        return f"Error updating profile: {str(e)}"
    finally:
        cursor.close()
        conn.close()

@tool
def parse_track_selection(config: RunnableConfig, selected_track_ids: list[int] = None, selected_tracks: list[str] = None) -> dict:
    """Get details for the selected tracks to prepare for purchase.
    
    Args:
        config: Configuration containing track information
        selected_track_ids: List of track IDs selected by the user
        selected_tracks: Backup list of track names/descriptions to search for if IDs fail
    
    Returns:
        A dictionary containing the selected track IDs and preview
    """
    try:
        # Get track info from messages history first
        messages = config.get("messages", [])
        track_info = None
        
        # Look for the most recent get_recommendations response
        for message in reversed(messages):
            # Try multiple ways to extract track info from message history
            if isinstance(message, dict):
                # Check if this is a tool message with get_recommendations
                if message.get("name") == "get_recommendations":
                    content = message.get("content")
                    if isinstance(content, str):
                        import json
                        try:
                            content_dict = json.loads(content)
                            track_info = content_dict.get("track_info", {})
                            if track_info:
                                break
                        except json.JSONDecodeError:
                            continue
                
                # Also check for tool_calls that might have recommendations
                tool_calls = message.get("tool_calls", [])
                if tool_calls and isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and tool_call.get("name") == "get_recommendations":
                            try:
                                content = tool_call.get("content")
                                if content and isinstance(content, str):
                                    content_dict = json.loads(content)
                                    track_info = content_dict.get("track_info", {})
                                    if track_info:
                                        break
                            except (json.JSONDecodeError, AttributeError):
                                continue
        
        # If we still don't have track info, try parsing the raw content of any message
        if not track_info:
            for message in reversed(messages):
                content = None
                if isinstance(message, dict):
                    content = message.get("content")
                if content and isinstance(content, str) and "{" in content and "track_info" in content:
                    try:
                        # Try to extract JSON from the content
                        import re
                        json_match = re.search(r'(\{.*"track_info":.+\})', content, re.DOTALL)
                        if json_match:
                            potential_json = json_match.group(1)
                            try:
                                content_dict = json.loads(potential_json)
                                track_info = content_dict.get("track_info", {})
                                if track_info:
                                    break
                            except json.JSONDecodeError:
                                continue
                    except Exception:
                        continue
        
        selected_tracks_info = []
        
        # Try track IDs first if provided
        if selected_track_ids and track_info:
            for track in track_info.values():
                if track["track_id"] in selected_track_ids:
                    selected_tracks_info.append(track)
        
        # If no tracks found and we have track names, try semantic search
        if not selected_tracks_info and selected_tracks:
            try:
                vector_store = get_vector_store()
                for track_name in selected_tracks:
                    # Search for each track individually to get best matches
                    docs = vector_store.similarity_search(track_name, k=1)
                    if docs:
                        doc = docs[0]
                        track = {
                            "track_id": int(doc.metadata['track_id']),
                            "track_name": doc.metadata['track_name'],
                            "artist": doc.metadata['artist'],
                            "album": doc.metadata['album'],
                            "genre": doc.metadata['genre'],
                            "price": float(doc.metadata['price'])
                        }
                        selected_tracks_info.append(track)
            except Exception as e:
                logger.error(f"Error in semantic search fallback: {str(e)}")
        
        # If still no tracks found, fall back to using just the track IDs directly
        if not selected_tracks_info and selected_track_ids:
            try:
                import sqlite3
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                for track_id in selected_track_ids:
                    cursor.execute("""
                        SELECT t.TrackId, t.Name as track_name, ar.Name as artist, 
                               al.Title as album, g.Name as genre, t.UnitPrice as price
                        FROM Track t
                        JOIN Album al ON t.AlbumId = al.AlbumId
                        JOIN Artist ar ON al.ArtistId = ar.ArtistId
                        JOIN Genre g ON t.GenreId = g.GenreId
                        WHERE t.TrackId = ?
                    """, (track_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        track = {
                            "track_id": row["TrackId"],
                            "track_name": row["track_name"],
                            "artist": row["artist"], 
                            "album": row["album"],
                            "genre": row["genre"],
                            "price": float(row["price"])
                        }
                        selected_tracks_info.append(track)
                
                cursor.close()
                conn.close()
            except Exception as e:
                logger.error(f"Error in database fallback: {str(e)}")
        
        if not selected_tracks_info:
            return {
                "status": "error",
                "message": "Could not find the selected tracks. Please try searching again or specify the tracks more clearly."
            }
        
        # Format the preview and prepare track IDs for purchase
        track_ids = [track["track_id"] for track in selected_tracks_info]
        total = sum(float(track['price']) for track in selected_tracks_info)
        
        return {
            "status": "success",
            "message": "Selected tracks for purchase:\n" + "\n".join(
                f"- {track['track_name']} by {track['artist']} (${float(track['price']):.2f})"
                for track in selected_tracks_info
            ) + f"\n\nTotal: ${total:.2f}",
            "track_ids": track_ids
        }
        
    except Exception as e:
        logger.error(f"Error parsing track selection: {str(e)}")
        return {
            "status": "error",
            "message": f"Error parsing track selection: {str(e)}"
        }

def get_recommendations_fallback(query: str, config: RunnableConfig) -> dict:
    """Fallback method for getting recommendations when vector store isn't available.
    Uses direct database queries based on keywords in the query."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Extract potential keywords from the query
        # This is a simple keyword extraction, not as good as vector search
        keywords = re.findall(r'\b\w+\b', query.lower())
        genre_keywords = []
        artist_keywords = []
        album_keywords = []
        
        # Common genre words that might appear in queries
        common_genres = ["rock", "pop", "jazz", "blues", "classical", "metal", 
                        "punk", "hip hop", "rap", "country", "folk", "electronic", 
                        "dance", "reggae", "indie", "alternative"]
        
        for word in keywords:
            if word in common_genres:
                genre_keywords.append(word)
            elif len(word) > 3:  # Only consider words with more than 3 chars for artist/album search
                artist_keywords.append(word)
                album_keywords.append(word)
        
        # Start building the query
        sql_query = """
            SELECT 
                t.TrackId, t.Name as track_name, ar.Name as artist, 
                al.Title as album, g.Name as genre, t.UnitPrice as price
            FROM Track t
            JOIN Album al ON t.AlbumId = al.AlbumId
            JOIN Artist ar ON al.ArtistId = ar.ArtistId
            JOIN Genre g ON t.GenreId = g.GenreId
            WHERE 1=1
        """
        params = []
        
        # Add genre filters if relevant
        if genre_keywords:
            genre_clauses = []
            for genre in genre_keywords:
                genre_clauses.append("g.Name LIKE ?")
                params.append(f"%{genre}%")
            
            if genre_clauses:
                sql_query += " AND (" + " OR ".join(genre_clauses) + ")"
        
        # Add artist filters if relevant
        if artist_keywords and not genre_keywords:  # Only apply if no genre filters
            artist_clauses = []
            for artist in artist_keywords:
                artist_clauses.append("ar.Name LIKE ?")
                params.append(f"%{artist}%")
            
            if artist_clauses:
                sql_query += " AND (" + " OR ".join(artist_clauses) + ")"
        
        # Add album filters if no other filters
        if album_keywords and not genre_keywords and not artist_keywords:
            album_clauses = []
            for album in album_keywords:
                album_clauses.append("al.Title LIKE ?")
                params.append(f"%{album}%")
            
            if album_clauses:
                sql_query += " AND (" + " OR ".join(album_clauses) + ")"
        
        # Limit to 10 random results
        sql_query += " ORDER BY RANDOM() LIMIT 10"
        
        cursor.execute(sql_query, params)
        rows = cursor.fetchall()
        
        if not rows:
            # If no results, return a general sample
            cursor.execute("""
                SELECT 
                    t.TrackId, t.Name as track_name, ar.Name as artist, 
                    al.Title as album, g.Name as genre, t.UnitPrice as price
                FROM Track t
                JOIN Album al ON t.AlbumId = al.AlbumId
                JOIN Artist ar ON al.ArtistId = ar.ArtistId
                JOIN Genre g ON t.GenreId = g.GenreId
                ORDER BY RANDOM()
                LIMIT 10
            """)
            rows = cursor.fetchall()
        
        tracks = []
        message_lines = [f"Here are some recommendations based on your search for '{query}':"]
        
        for i, row in enumerate(rows, 1):
            track = {
                "track_id": row["TrackId"],
                "track_name": row["track_name"],
                "artist": row["artist"],
                "album": row["album"],
                "genre": row["genre"],
                "price": float(row["price"])
            }
            tracks.append(track)
            
            message_lines.append(
                f"{i}. {track['track_name']} by {track['artist']} "
                f"(Album: {track['album']}, Genre: {track['genre']}, "
                f"Price: ${track['price']:.2f})"
            )
        
        cursor.close()
        conn.close()
        
        # Return structured response
        return {
            "status": "success",
            "message": "\n".join(message_lines),
            "track_info": {str(i): track for i, track in enumerate(tracks, 1)},
            "track_ids": [track["track_id"] for track in tracks]
        }
        
    except Exception as e:
        logger.error(f"Error in get_recommendations_fallback: {str(e)}")
        return {
            "status": "error",
            "message": f"Sorry, I couldn't find any music recommendations at the moment. The music catalog seems to be unavailable. Please try again later."
        }

    