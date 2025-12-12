#!/usr/bin/env python3
"""
Reset script - Clears all data and re-ingests documents from scratch.

This script:
1. Clears Qdrant vector database (recreates collection)
2. Clears Graphiti/FalkorDB knowledge graph (restarts container to clear data)
3. Resets sources_state.json
4. Re-ingests all documents
"""
import asyncio
import json
import httpx
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from ingest import ingest_all_documents, list_document_ids, COLLECTION_NAME, VECTOR_SIZE

STATE_FILE = Path("sources_state.json")


async def clear_graphiti_all():
    """Clear all data from Graphiti/FalkorDB by calling clear_graph with all group IDs."""
    print("\n🗑️  Clearing Graphiti/FalkorDB knowledge graph...")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # Initialize session
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
                        "clientInfo": {"name": "reset-script", "version": "0.1"}
                    },
                    "id": 1
                }
            )
            
            session_id = init.headers.get("mcp-session-id")
            if not session_id:
                print("⚠️  Could not get Graphiti session ID. Attempting to clear anyway...")
                # Try to clear by restarting container instead
                return await restart_graphiti_container()
            
            # Get all document IDs to clear their groups
            doc_ids = list_document_ids()
            
            # Sanitize group IDs (same logic as in ingest.py)
            import re
            group_ids = [re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id) for doc_id in doc_ids]
            
            if not group_ids:
                print("  No groups to clear.")
                return
            
            print(f"  Clearing {len(group_ids)} groups from Graphiti...")
            
            # Call clear_graph for all groups
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
                        "name": "clear_graph",
                        "arguments": {
                            "group_ids": group_ids
                        }
                    },
                    "id": 2
                }
            )
            
            print(f"  ✓ Graphiti clear_graph response: {response.status_code}")
            
        except Exception as e:
            print(f"  ⚠️  Error clearing Graphiti via API: {e}")
            print("  Attempting container restart instead...")
            await restart_graphiti_container()


async def restart_graphiti_container():
    """Restart Graphiti container to clear all data."""
    import subprocess
    print("  Restarting Graphiti container to clear all data...")
    try:
        result = subprocess.run(
            ["docker", "restart", "docker-graphiti-mcp-1"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print("  ✓ Graphiti container restarted")
            # Wait a moment for it to be ready
            await asyncio.sleep(3)
        else:
            print(f"  ⚠️  Container restart returned: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️  Error restarting container: {e}")
        print("  You may need to manually restart: docker restart docker-graphiti-mcp-1")


def clear_qdrant():
    """Clear Qdrant vector database by recreating the collection."""
    print("\n🗑️  Clearing Qdrant vector database...")
    try:
        qdrant = QdrantClient(host="localhost", port=6333)
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("  ✓ Qdrant collection recreated (all data cleared)")
    except Exception as e:
        print(f"  ✗ Error clearing Qdrant: {e}")
        raise


def reset_sources_state():
    """Reset sources_state.json to empty state."""
    print("\n🗑️  Resetting sources_state.json...")
    state = {"sources": {}}
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print("  ✓ sources_state.json reset")


async def main():
    """Main reset and re-ingest process."""
    print("=" * 60)
    print("🔄 RESETTING ALL DATA AND RE-INGESTING DOCUMENTS")
    print("=" * 60)
    
    # Step 1: Clear Qdrant
    clear_qdrant()
    
    # Step 2: Clear Graphiti/FalkorDB
    await clear_graphiti_all()
    
    # Step 3: Reset sources_state.json
    reset_sources_state()
    
    # Step 4: Re-ingest all documents
    print("\n" + "=" * 60)
    print("📥 RE-INGESTING ALL DOCUMENTS")
    print("=" * 60)
    await ingest_all_documents()
    
    print("\n" + "=" * 60)
    print("✅ RESET AND RE-INGESTION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())



