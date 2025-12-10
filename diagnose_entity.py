#!/usr/bin/env python3
"""
Diagnostic script to check entity extraction and search issues.
Checks what Graphiti has for "Raymond Labarge" and why search isn't finding it.
"""
import asyncio
import httpx
from qdrant_client import QdrantClient
from ingest import list_document_ids

async def check_graphiti_entity(entity_name: str):
    """Check what Graphiti knows about an entity."""
    print(f"\n🔍 Checking Graphiti for: '{entity_name}'")
    print("=" * 60)
    
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
                        "clientInfo": {"name": "diagnostic", "version": "0.1"}
                    },
                    "id": 1
                }
            )
            
            session_id = init.headers.get("mcp-session-id")
            if not session_id:
                print("❌ Could not get Graphiti session ID")
                return
            
            print(f"✓ Connected to Graphiti (session: {session_id[:8]}...)")
            
            # Search for the entity
            print(f"\n📊 Searching Graphiti for: '{entity_name}'...")
            search_response = await client.post(
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
                        "name": "search",
                        "arguments": {
                            "query": entity_name,
                            "num_results": 10
                        }
                    },
                    "id": 2
                }
            )
            
            response_text = search_response.text
            print(f"\n📄 Graphiti search response (first 500 chars):")
            print(response_text[:500])
            
            # Try to get nodes
            print(f"\n🔎 Getting nodes for: '{entity_name}'...")
            nodes_response = await client.post(
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
                        "name": "graph_query",
                        "arguments": {
                            "query": f"MATCH (n) WHERE n.name CONTAINS '{entity_name}' OR n.label CONTAINS '{entity_name}' RETURN n"
                        }
                    },
                    "id": 3
                }
            )
            
            nodes_text = nodes_response.text
            print(f"\n📊 Graphiti nodes response (first 500 chars):")
            print(nodes_text[:500])
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


def check_qdrant_search(entity_name: str):
    """Check Qdrant vector search for the entity."""
    print(f"\n🔍 Checking Qdrant vector search for: '{entity_name}'")
    print("=" * 60)
    
    try:
        qdrant = QdrantClient(host="localhost", port=6333)
        
        # Create embedding for the entity name
        from openai import OpenAI
        openai_client = OpenAI()
        
        embedding = openai_client.embeddings.create(
            input=entity_name,
            model="text-embedding-3-small"
        ).data[0].embedding
        
        # Search
        results = qdrant.query_points(
            collection_name="archive_documents",
            query=embedding,
            limit=5,
            with_payload=True
        )
        
        print(f"✓ Found {len(results.points)} results in Qdrant:")
        for i, hit in enumerate(results.points, 1):
            doc_id = hit.payload.get("doc_id", "unknown")
            chunk_id = hit.payload.get("chunk_id", "?")
            score = hit.score
            text_preview = hit.payload.get("text", "")[:100]
            
            print(f"\n  [{i}] Document: {doc_id} (chunk {chunk_id})")
            print(f"      Relevance: {score:.3f}")
            print(f"      Preview: {text_preview}...")
            
            # Check if entity name appears in text
            text = hit.payload.get("text", "").lower()
            if entity_name.lower() in text:
                print(f"      ✓ Entity name FOUND in this chunk")
            else:
                print(f"      ✗ Entity name NOT found in this chunk")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def check_document_text(entity_name: str):
    """Check which documents actually contain this entity."""
    print(f"\n📄 Checking documents for: '{entity_name}'")
    print("=" * 60)
    
    doc_ids = list_document_ids()
    found_in = []
    
    for doc_id in doc_ids:
        try:
            from ingest import read_document_text
            text = read_document_text(doc_id)
            if entity_name.lower() in text.lower():
                found_in.append(doc_id)
                # Find line number
                lines = text.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if entity_name.lower() in line.lower():
                        print(f"✓ Found in '{doc_id}' at line {line_num}")
                        print(f"  Context: {line.strip()[:100]}...")
                        break
        except Exception as e:
            print(f"⚠️  Error reading {doc_id}: {e}")
    
    if not found_in:
        print(f"✗ '{entity_name}' NOT FOUND in any document")
    else:
        print(f"\n✓ Found in {len(found_in)} document(s): {', '.join(found_in)}")


async def main():
    entity_name = "Raymond Labarge"
    
    print("=" * 60)
    print("DIAGNOSTIC: Raymond Labarge Entity Investigation")
    print("=" * 60)
    
    # Check documents first
    check_document_text(entity_name)
    
    # Check Qdrant search
    check_qdrant_search(entity_name)
    
    # Check Graphiti
    await check_graphiti_entity(entity_name)
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


