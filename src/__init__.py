"""
Music Store Support Bot - A LangGraph-based conversational agent for music recommendations
"""

from .agents.music_store_agent import graph
from .config.settings import setup_langsmith, check_api_keys
from .tools import (
    query_invoice_history,
    get_recommendations,
    fetch_customer_info,
    request_refund,
    process_purchase,
    update_profile
)

__version__ = "0.1.0"
__all__ = [
    "graph",
    "setup_langsmith",
    "check_api_keys",
    "query_invoice_history",
    "get_recommendations",
    "fetch_customer_info",
    "request_refund",
    "process_purchase",
    "update_profile"
] 