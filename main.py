# main.py - FastAPI backend
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Clients
openai_client = OpenAI()
qdrant = QdrantClient(host="localhost", port=6333)

class QueryRequest(BaseModel):
    question: str
    user_id: str = "researcher_1"

async def query_graphiti(question: str):
    """Query Graphiti knowledge graph"""
    async with httpx.AsyncClient(timeout=30.0) as client:
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
                    "name": "search",
                    "arguments": {
                        "query": question,
                        "num_results": 10
                    }
                },
                "id": 1
            }
        )
        return response.text  # SSE format, we'll extract what we can

def query_qdrant(question: str, user_permissions: list):
    """Query Qdrant vector database"""
    # Embed question
    embedding = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search with governance filter
    results = qdrant.query_points(
        collection_name="archive_documents",
        query=embedding,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="access_level",
                    match=MatchAny(any=user_permissions)
                )
            ]
        ),
        limit=5,
        with_payload=True
    )

    return results.points

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Main query endpoint"""
    try:
        # User permissions (hardcoded for PoC)
        permissions = ["public", "researcher"]

        # 1. Query Graphiti (knowledge graph)
        print(f"Querying Graphiti: {request.question}")
        graph_response = await query_graphiti(request.question)

        # 2. Query Qdrant (document chunks)
        print(f"Querying Qdrant: {request.question}")
        doc_results = query_qdrant(request.question, permissions)

        # 3. Build combined context
        context_parts = []

        # Add graph context
        context_parts.append("KNOWLEDGE GRAPH INSIGHTS:")
        context_parts.append(f"{graph_response[:500]}")  # First 500 chars
        context_parts.append("")

        # Add document context
        context_parts.append("RELEVANT DOCUMENTS:")
        for i, hit in enumerate(doc_results, 1):
            context_parts.append(f"\n[Document {i}: {hit.payload['title']}]")
            context_parts.append(f"{hit.payload['text'][:800]}")  # First 800 chars

        context = "\n".join(context_parts)

        # 4. Generate answer with OpenAI
        print("Generating answer with OpenAI...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an archivist assistant for Indigenous historical documents.

RULES:
1. Always cite documents with their titles
2. When sources contradict, acknowledge explicitly
3. Use temporal language when relevant
4. Distinguish official vs. community perspectives
5. Be honest about uncertainty"""
                },
                {
                    "role": "user",
                    "content": f"""CONTEXT:
{context}

QUESTION: {request.question}

Provide a comprehensive answer with citations to document titles. If the context doesn't contain relevant information, say so."""
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        # 5. Return structured response
        return {
            "answer": answer,
            "sources": [
                {
                    "title": hit.payload["title"],
                    "relevance": hit.score
                }
                for hit in doc_results
            ],
            "graph_context": graph_response[:200]  # Preview
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "graph_context": ""
        }

@app.get("/")
def root():
    """Serve frontend"""
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)