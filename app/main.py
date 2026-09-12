"""
main.py
The FastAPI application. Run this with:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs (auto-generated
by FastAPI -- great for testing without Postman).
"""
from fastapi import FastAPI, HTTPException
from app.database import get_connection, init_db
from app.models import NoteCreate, NoteResponse, SearchResult
from app.search import search_notes

app = FastAPI(title="Study Notes API", version="1.0")


@app.on_event("startup")
def on_startup():
    """Create the database table when the server starts."""
    init_db()


@app.post("/notes", response_model=NoteResponse)
def create_note(note: NoteCreate):
    """Add a new note to the database."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO notes (title, content, tags) VALUES (?, ?, ?)",
        (note.title, note.content, note.tags),
    )
    conn.commit()
    new_id = cursor.lastrowid

    row = conn.execute("SELECT * FROM notes WHERE id = ?", (new_id,)).fetchone()
    conn.close()
    return dict(row)


@app.get("/notes", response_model=list[NoteResponse])
def list_notes():
    """Return all notes, most recent first."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/notes/{note_id}", response_model=NoteResponse)
def get_note(note_id: int):
    """Return a single note by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return dict(row)


@app.delete("/notes/{note_id}")
def delete_note(note_id: int):
    """Delete a note by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Note not found")

    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()
    return {"message": f"Note {note_id} deleted"}


@app.get("/search", response_model=list[SearchResult])
def search(q: str, top_k: int = 5):
    """
    Search notes by keyword relevance using TF-IDF.
    Example: GET /search?q=normalization
    """
    conn = get_connection()
    rows = conn.execute("SELECT id, title, content FROM notes").fetchall()
    conn.close()

    notes = [dict(row) for row in rows]
    results = search_notes(q, notes, top_k=top_k)
    return results
