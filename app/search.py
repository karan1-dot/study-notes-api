"""
search.py
Implements keyword search over notes using TF-IDF (Term Frequency - Inverse
Document Frequency) and cosine similarity -- built from scratch using only
Python's standard library (no scikit-learn/numpy needed).

WHY TF-IDF (explain this in your interview):
- It's a classic NLP technique used to rank documents by relevance to a query.
- Instead of a plain SQL "LIKE" match, TF-IDF weighs words by how important
  they are: common words (the, is, and) get low weight, rare/specific words
  get high weight. This means a search for "gradient descent" will rank a
  note that's actually ABOUT gradient descent higher than a note that just
  mentions it in passing.
- This directly demonstrates "foundational AI/ML concepts" and gives you a
  genuine talking point about vectorization and similarity scoring -- and
  since it's hand-built here, you can explain every step, not just "I called
  a library function."

HOW IT WORKS, STEP BY STEP:
1. Tokenize: split each note's text into lowercase words.
2. Term Frequency (TF): how often each word appears in a given document.
3. Inverse Document Frequency (IDF): words that appear in MANY documents are
   less informative (e.g. "the"), so they get a lower weight. Rare words that
   appear in few documents get a higher weight.
4. TF-IDF score for a word in a document = TF * IDF.
5. Represent each document (and the query) as a vector of TF-IDF scores.
6. Cosine similarity measures the angle between two vectors -- closer to 1
   means more similar. This ranks notes by relevance to the query.
"""
import math
from collections import Counter

# A small set of common English words to ignore, so they don't dominate
# scores just because they appear everywhere.
STOP_WORDS = {
    "the", "is", "in", "at", "of", "a", "an", "and", "or", "to", "for",
    "on", "with", "by", "it", "this", "that", "as", "are", "was", "be",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into words, stripping simple punctuation."""
    words = text.lower().split()
    cleaned = [w.strip(".,!?;:()[]\"'") for w in words]
    return [w for w in cleaned if w and w not in STOP_WORDS]


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Term Frequency: fraction of the document each word makes up."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {word: count / total for word, count in counts.items()}


def _compute_idf(all_tokenized_docs: list[list[str]]) -> dict[str, float]:
    """Inverse Document Frequency across the whole corpus (all notes + query)."""
    num_docs = len(all_tokenized_docs)
    doc_freq = Counter()
    for tokens in all_tokenized_docs:
        for word in set(tokens):
            doc_freq[word] += 1

    # +1 smoothing avoids division by zero and log(1)=0 issues for words
    # that appear in every document.
    return {
        word: math.log((num_docs + 1) / (freq + 1)) + 1
        for word, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Combine TF and IDF into a single TF-IDF vector for one document."""
    tf = _compute_tf(tokens)
    return {word: tf_val * idf.get(word, 0.0) for word, tf_val in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    shared_words = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[w] * vec_b[w] for w in shared_words)

    magnitude_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    magnitude_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def search_notes(query: str, notes: list[dict], top_k: int = 5):
    """
    Ranks `notes` by relevance to `query` using hand-built TF-IDF + cosine similarity.

    Args:
        query: the search string typed by the user
        notes: list of dicts, each with at least 'id', 'title', 'content'
        top_k: how many top results to return

    Returns:
        list of dicts with an added 'score' field, sorted by relevance (desc)
    """
    if not notes:
        return []

    # Tokenize every note and the query using the same vocabulary space
    note_tokens = [_tokenize(f"{n['title']} {n['content']}") for n in notes]
    query_tokens = _tokenize(query)

    all_tokenized = note_tokens + [query_tokens]
    idf = _compute_idf(all_tokenized)

    query_vector = _tfidf_vector(query_tokens, idf)

    scored_notes = []
    for note, tokens in zip(notes, note_tokens):
        note_vector = _tfidf_vector(tokens, idf)
        score = _cosine_similarity(query_vector, note_vector)
        if score > 0:  # skip completely irrelevant notes
            scored_notes.append({**note, "score": score})

    scored_notes.sort(key=lambda n: n["score"], reverse=True)
    return scored_notes[:top_k]
