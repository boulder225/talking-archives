# ingest.py
import asyncio
import re
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

DOCUMENTS_DIR = Path("documents")
COLLECTION_NAME = "archive_documents"
VECTOR_SIZE = 1536
MAX_CHARS_PER_EPISODE = 4000
TARGET_WORDS_PER_CHUNK = 500
MAX_CHARS_PER_CHUNK = 3000
QDRANT_BATCH_SIZE = 50


def create_openai_client():
    return OpenAI()


def create_qdrant_client():
    return QdrantClient(host="localhost", port=6333)


def ensure_qdrant_collection(qdrant_client: QdrantClient, recreate: bool = False):
    if recreate:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        return

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )


def count_words(text: str) -> int:
    """Count words in text. Simple word counter using whitespace."""
    if not text or not text.strip():
        return 0
    # Split by whitespace and count non-empty segments
    return len([word for word in text.split() if word.strip()])


def extract_mentions(text: str, max_mentions: int = 40) -> List[str]:
    """Naive proper-noun extraction for tagging."""
    if not text:
        return []
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", text)
    seen = set()
    mentions = []
    for cand in candidates:
        cleaned = cand.strip()
        if len(cleaned) < 4:
            continue
        if cleaned.lower() in seen:
            continue
        mentions.append(cleaned)
        seen.add(cleaned.lower())
        if len(mentions) >= max_mentions:
            break
    return mentions


def chunk_text(
    text: str,
    target_words: int = TARGET_WORDS_PER_CHUNK,
    max_chars: int = MAX_CHARS_PER_CHUNK
) -> List[str]:
    """
    Split text into ~target_words chunks, respecting paragraph boundaries.

    Uses word-based counting for consistent semantic units across documents.
    Hard cap: max_chars to prevent overly long segments.
    Always respects paragraph boundaries to maintain semantic coherence.
    """
    if not text or not text.strip():
        return [text.strip()] if text else []

    # Split into paragraphs (preserve double newlines)
    paragraphs = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            paragraphs.append(para)

    # If no paragraph breaks, split by single newlines
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # If still no breaks, treat entire text as one paragraph
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current_paragraphs = []
    current_word_count = 0
    current_char_count = 0

    for para in paragraphs:
        para_word_count = count_words(para)
        para_char_count = len(para)

        # Check if adding this paragraph would exceed limits
        would_exceed_words = current_word_count + para_word_count > target_words
        would_exceed_chars = current_char_count + para_char_count > max_chars

        # If we have content and adding this para would exceed limits, finalize chunk
        if current_paragraphs and (would_exceed_words or would_exceed_chars):
            chunk_text = "\n\n".join(current_paragraphs).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_paragraphs = []
            current_word_count = 0
            current_char_count = 0

        # Add paragraph to current chunk
        current_paragraphs.append(para)
        current_word_count += para_word_count
        current_char_count += para_char_count

        # If a single paragraph exceeds max_chars, we must include it anyway
        if para_char_count > max_chars:
            pass  # Include it whole to avoid breaking semantic coherence

    # Add remaining paragraphs as final chunk
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Fallback: if somehow we have no chunks, return original text
    if not chunks:
        chunks.append(text.strip())

    return chunks


async def ingest_chunks_to_graphiti(base_name: str, chunks: List[str], date: str, group_id: str):
    """Send multiple chunks of a document to Graphiti."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        init = await client.post(
            "http://localhost:8000/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "talking-archives-ingest", "version": "0.1"}
                },
                "id": 1
            }
        )

        session_id = init.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError("Graphiti MCP initialize response missing session ID")

        for idx, chunk in enumerate(chunks, start=1):
            chunk_name = f"{base_name} (Part {idx})" if len(chunks) > 1 else base_name
            episode_content = f"[Reference Time: {date}]\n{chunk}"

            await client.post(
                "http://localhost:8000/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Session-Id": session_id
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "add_memory",
                        "arguments": {
                            "name": chunk_name,
                            "episode_body": episode_content,
                            "source": "text",
                            "source_description": f"Talking Archives ingestion - {base_name}",
                            "group_id": group_id
                        }
                    },
                    "id": idx + 1
                }
            )


def list_document_ids() -> List[str]:
    return sorted(p.stem for p in DOCUMENTS_DIR.glob("*.txt"))


def qdrant_point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"talking-archives/{doc_id}"))


async def ingest_document(
    doc_id: str,
    openai_client: Optional[OpenAI] = None,
    qdrant_client: Optional[QdrantClient] = None,
    recreate_collection: bool = False,
    reference_time: str = "1974-01-01T00:00:00Z"
):
    openai_client = openai_client or create_openai_client()
    qdrant_client = qdrant_client or create_qdrant_client()

    ensure_qdrant_collection(qdrant_client, recreate=recreate_collection)

    doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
    if not doc_path.exists():
        raise FileNotFoundError(f"Document {doc_id} not found in documents/")

    text = doc_path.read_text()
    if len(text.strip()) < 50:
        raise ValueError("Document too short to ingest")

    chunks = chunk_text(text)
    # Sanitize group_id for Graphiti (same logic used elsewhere)
    import re
    sanitized_group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    await ingest_chunks_to_graphiti(
        base_name=doc_id,
        chunks=chunks,
        date=reference_time,
        group_id=sanitized_group_id
    )

    # Create separate Qdrant points for each chunk
    points = []
    for i, chunk in enumerate(chunks):
        embedding = openai_client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding

        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_chunk_{i}")),
            vector=embedding,
            payload={
                "text": chunk,
                "title": doc_id,
                "doc_id": doc_id,
                "chunk_id": i,
                "word_count": count_words(chunk),
                "char_count": len(chunk),
                "access_level": "public",
                "mentions": extract_mentions(chunk)
            }
        ))

    # Batch upsert to Qdrant
    print(f"  Upserting {len(points)} points to Qdrant in batches of {QDRANT_BATCH_SIZE}...")
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch_end = min(batch_start + QDRANT_BATCH_SIZE, len(points))
        batch = points[batch_start:batch_end]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"    Upserted batch {batch_start // QDRANT_BATCH_SIZE + 1} ({batch_start + 1}-{batch_end}/{len(points)})...")
    print(f"  ✓ All {len(points)} chunks upserted to Qdrant")


async def ingest_all_documents(
    openai_client: Optional[OpenAI] = None,
    qdrant_client: Optional[QdrantClient] = None
):
    openai_client = openai_client or create_openai_client()
    qdrant_client = qdrant_client or create_qdrant_client()

    ensure_qdrant_collection(qdrant_client, recreate=True)

    for doc_id in list_document_ids():
        print(f"Ingesting {doc_id}...")
        await ingest_document(
            doc_id,
            openai_client=openai_client,
            qdrant_client=qdrant_client,
            recreate_collection=False
        )
        print(f"✓ {doc_id} complete")


if __name__ == "__main__":
    print("Starting ingestion...\n")
    asyncio.run(ingest_all_documents())
    print("\n✅ Ingestion complete!")