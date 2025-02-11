from flask import Flask, request, jsonify
from src.agents.music_store_agent import get_agent_response
from src.config.settings import check_api_keys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Check for required API keys on startup
check_api_keys()

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        message = data.get('message')
        customer_id = data.get('customer_id')
        thread_id = data.get('thread_id')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        if not customer_id:
            return jsonify({"error": "No customer_id provided"}), 400
            
        # Get response from agent
        messages, new_thread_id = get_agent_response(message, customer_id, thread_id)
        
        # Format response
        response = {
            "messages": [
                {
                    "role": msg.type if hasattr(msg, 'type') else "unknown",
                    "content": msg.content
                }
                for msg in messages
            ],
            "thread_id": new_thread_id
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 