"""
models.py
Pydantic models define the shape of data going in and out of the API.
FastAPI uses these to auto-validate requests and auto-generate docs.
"""
from pydantic import BaseModel
from typing import Optional


class NoteCreate(BaseModel):
    """What the client sends when creating a note."""
    title: str
    content: str
    tags: Optional[str] = None  # comma-separated tags, e.g. "dbms,sql"


class NoteResponse(BaseModel):
    """What the API sends back for a single note."""
    id: int
    title: str
    content: str
    tags: Optional[str]
    created_at: str


class SearchResult(BaseModel):
    """A single search hit, with a relevance score from TF-IDF."""
    id: int
    title: str
    content: str
    score: float
