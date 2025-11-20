# ingest.py
import asyncio
import os
from pathlib import Path
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import httpx
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

# Create Qdrant collection
print("Creating Qdrant collection...")
qdrant.recreate_collection(
    collection_name="archive_documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
print("✓ Collection created\n")

async def ingest_to_graphiti(name: str, text: str, date: str):
    """Ingest document to Graphiti via HTTP"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "http://localhost:8000/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add_episode",
                    "arguments": {
                        "name": name,
                        "episode_body": text[:8000],  # Limit size
                        "reference_time": date,
                        "source": "text"
                    }
                },
                "id": 1
            }
        )
        return response.status_code

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

            # 1. Ingest to Graphiti (entity extraction)
            print(f"  → Sending to Graphiti...")
            status = await ingest_to_graphiti(
                name=txt_file.stem,
                text=text,
                date="1974-01-01T00:00:00Z"  # Adjust based on your docs
            )
            print(f"  ✓ Graphiti: {status}")

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
                        "access_level": "public"
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