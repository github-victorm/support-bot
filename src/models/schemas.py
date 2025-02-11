from pydantic import BaseModel, Field

class CustomerProfile(BaseModel):
    """Profile information about the customer"""
    favorite_genres: list[str] = Field(description="List of music genres the customer enjoys", default_factory=list)
    favorite_artists: list[str] = Field(description="List of artists the customer enjoys", default_factory=list)
    recent_purchases: list[str] = Field(description="List of recent music purchases", default_factory=list)
    listening_preferences: list[str] = Field(description="General preferences about music listening habits", default_factory=list) 