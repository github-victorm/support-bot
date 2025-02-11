import os
from dotenv import load_dotenv
import logging
from langsmith import Client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_langsmith():
    """Initialize LangSmith for tracking"""
    if os.getenv("LANGSMITH_API_KEY"):
        try:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "music-store-support"
            os.environ["LANGCHAIN_TAGS"] = "production" if os.getenv("PRODUCTION") else "development"
            os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"
            client = Client()
            logger.info("LangSmith tracking enabled")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize LangSmith: {str(e)}")
            return None
    else:
        logger.warning("LANGSMITH_API_KEY not found. Tracing disabled.")
        return None

def check_api_keys():
    """Check for required API keys"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables") 