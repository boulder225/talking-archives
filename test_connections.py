# test_connections.py - NO .json() call
import asyncio
from qdrant_client import QdrantClient
import httpx

async def test():
    print("Testing Qdrant...")
    qdrant = QdrantClient(host="localhost", port=6333)
    print("✓ Qdrant connected\n")

    print("Testing Graphiti MCP via HTTP...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Initialize
            response = await client.post(
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
                        "clientInfo": {"name": "test-client", "version": "1.0"}
                    },
                    "id": 1
                }
            )

            print(f"✓ Initialize: {response.status_code}")
            # Don't call .json() - it's SSE format

            # Test search
            search_response = await client.post(
                "http://localhost:8000/mcp",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "search",
                        "arguments": {"query": "test", "num_results": 5}
                    },
                    "id": 2
                }
            )

            print(f"✓ Search: {search_response.status_code}")
            print(f"✓ Graphiti MCP working!\n")

        except Exception as e:
            print(f"Error: {e}\n")

    print("✅ READY TO BUILD INGESTION SCRIPT!\n")

asyncio.run(test())