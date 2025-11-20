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


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_EPISODE) -> List[str]:
    """Split text into roughly max_chars chunks, respecting paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 1  # include newline
        if current and current_len + para_len > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n".join(current).strip())

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
    await ingest_chunks_to_graphiti(
        base_name=doc_id,
        chunks=chunks,
        date=reference_time,
        group_id=doc_id
    )

    embedding = openai_client.embeddings.create(
        input=text[:8000],
        model="text-embedding-3-small"
    ).data[0].embedding

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=qdrant_point_id(doc_id),
            vector=embedding,
            payload={
                "text": text,
                "title": doc_id,
                "doc_id": doc_id,
                "access_level": "public",
                "mentions": extract_mentions(text)
            }
        )]
    )


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