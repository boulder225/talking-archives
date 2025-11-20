# ingest.py
import asyncio
import os
from pathlib import Path
from typing import List
import re
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import httpx
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

MAX_CHARS_PER_EPISODE = 4000


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

# Create Qdrant collection
print("Creating Qdrant collection...")
qdrant.recreate_collection(
    collection_name="archive_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print("✓ Collection created\n")

async def ingest_chunks_to_graphiti(base_name: str, chunks: List[str], date: str):
    """Send multiple chunks of a document to Graphiti."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Initialize session to get MCP-Session-Id
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

        statuses = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_name = f"{base_name} (Part {idx})" if len(chunks) > 1 else base_name
            episode_content = f"[Reference Time: {date}]\n{chunk}"

            response = await client.post(
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
                        }
                    },
                    "id": idx + 1
                }
            )
            statuses.append(response.status_code)

        return statuses

async def ingest_documents():
    doc_id = 0

    for txt_file in sorted(Path("documents").glob("*.txt")):
        doc_id += 1
        print(f"\n[{doc_id}] Processing {txt_file.name}...")

        try:
            text = txt_file.read_text()

            # Skip empty files
            if len(text.strip()) < 50:
                print(f"  ⚠ Skipping (too short)")
                continue

            # Split into Graphiti-friendly chunks
            chunks = chunk_text(text)

            # 1. Ingest to Graphiti (entity extraction)
            print(f"  → Sending to Graphiti ({len(chunks)} chunk(s))...")
            statuses = await ingest_chunks_to_graphiti(
                base_name=txt_file.stem,
                chunks=chunks,
                date="1974-01-01T00:00:00Z"  # Adjust based on your docs
            )
            print(f"  ✓ Graphiti: {len(statuses)} request(s)")

            # 2. Embed and ingest to Qdrant
            print(f"  → Embedding for Qdrant...")
            embedding = openai_client.embeddings.create(
                input=text[:8000],
                model="text-embedding-3-small"
            ).data[0].embedding

            qdrant.upsert(
                collection_name="archive_documents",
                points=[PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "title": txt_file.stem,
                        "doc_id": txt_file.stem,
                        "access_level": "public",
                        "mentions": extract_mentions(text)
                    }
                )]
            )
            print(f"  ✓ Qdrant: ingested")

            print(f"✅ {txt_file.name} complete")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

print("Starting ingestion...\n")
asyncio.run(ingest_documents())
print("\n✅ Ingestion complete!")