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
import json
import os
import re

load_dotenv()

app = FastAPI()

# Graphiti session tracking
graphiti_session_id = None
graphiti_session_lock = asyncio.Lock()

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
    """Query Graphiti knowledge graph for nodes, facts, and episodes."""
    nodes_struct, nodes_preview = await call_graphiti_tool(
        "search_nodes",
        {
            "query": question,
            "max_nodes": 10
        }
    )

    facts_struct, facts_preview = await call_graphiti_tool(
        "search_memory_facts",
        {
            "query": question,
            "max_facts": 10
        }
    )

    episodes_struct, episodes_preview = await call_graphiti_tool(
        "get_episodes",
        {
            "max_episodes": 5
        }
    )

    (
        graph_entities,
        graph_relationships,
        graph_episodes
    ) = normalize_graphiti_results(
        nodes_struct,
        facts_struct,
        episodes_struct
    )

    preview_parts = list(
        filter(
            None,
            [nodes_preview, facts_preview, episodes_preview]
        )
    )
    preview_text = "\n\n".join(preview_parts)

    return graph_entities, graph_relationships, graph_episodes, preview_text


async def ensure_graphiti_session():
    """Initialize Graphiti MCP session if needed."""
    global graphiti_session_id
    if graphiti_session_id:
        return graphiti_session_id

    async with graphiti_session_lock:
        if graphiti_session_id:
            return graphiti_session_id

        async with httpx.AsyncClient(timeout=15.0) as client:
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
                        "clientInfo": {"name": "talking-archives-backend", "version": "0.1"}
                    },
                    "id": 1
                }
            )

        session_id = response.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError("Graphiti initialize response missing session ID")

        graphiti_session_id = session_id
        return graphiti_session_id


async def reset_graphiti_session():
    """Reset cached session ID so the next call re-initializes."""
    global graphiti_session_id
    async with graphiti_session_lock:
        graphiti_session_id = None


async def call_graphiti_tool(name: str, arguments: dict):
    """Execute a Graphiti MCP tool call and return structured + preview text."""
    response_text = ""

    for _ in range(2):
        session_id = await ensure_graphiti_session()

        async with httpx.AsyncClient(timeout=30.0) as client:
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
                        "name": name,
                        "arguments": arguments
                    },
                    "id": 1
                }
            )

        response_text = response.text

        if "Missing session ID" in response_text:
            await reset_graphiti_session()
            continue

        structured, preview = parse_graphiti_tool_response(response_text)
        return structured, preview

    return {}, response_text


def parse_graphiti_tool_response(raw_text: str):
    """Parse SSE output from Graphiti tool call."""
    if not raw_text:
        return {}, ""

    structured = None
    text_chunks = []

    for line in raw_text.splitlines():
        stripped = line.strip()

        if not stripped or not stripped.startswith("data:"):
            continue

        payload_str = stripped[len("data:"):].strip()
        if payload_str in ("", "[DONE]"):
            continue

        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            continue

        result = payload.get("result", {})
        structured_content = result.get("structuredContent")
        if structured_content:
            structured = structured_content

        content_list = result.get("content") or []
        for item in content_list:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and item.get("text")
            ):
                text_chunks.append(item["text"])

    if structured is None and text_chunks:
        for chunk in reversed(text_chunks):
            try:
                structured = {"result": json.loads(chunk)}
                break
            except json.JSONDecodeError:
                continue

    preview_text = "\n\n".join(text_chunks).strip()
    return structured or {}, preview_text


def _unwrap_structured(structured):
    if not structured:
        return {}

    if isinstance(structured, dict) and "result" in structured:
        inner = structured.get("result")
        if isinstance(inner, dict):
            return inner
    return structured


def normalize_graphiti_results(nodes_struct, facts_struct, episodes_struct):
    """Convert Graphiti structured results into frontend-friendly lists."""
    entities = []
    relationships = []
    episodes = []

    nodes_data = _unwrap_structured(nodes_struct)
    for node in nodes_data.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        labels = node.get("labels") or []
        entity_type = labels[0] if labels else node.get("attributes", {}).get("type")
        entities.append({
            "name": node.get("name") or node.get("label"),
            "type": entity_type,
            "uuid": node.get("uuid")
        })

    facts_data = _unwrap_structured(facts_struct)
    for fact in facts_data.get("facts", []) or []:
        if not isinstance(fact, dict):
            continue

        source = fact.get("source") or {}
        target = fact.get("target") or {}

        rel_entry = {
            "from": (
                fact.get("source_name")
                or source.get("name")
                or fact.get("from")
            ),
            "to": (
                fact.get("target_name")
                or target.get("name")
                or fact.get("to")
            ),
            "type": (
                fact.get("relationship_type")
                or fact.get("type")
                or fact.get("relation")
                or fact.get("predicate")
            ),
            "valid_from": (
                fact.get("valid_from")
                or fact.get("start_time")
                or fact.get("startDate")
            ),
            "valid_to": (
                fact.get("valid_to")
                or fact.get("end_time")
                or fact.get("endDate")
            ),
            "fact": (
                fact.get("fact")
                or fact.get("summary")
                or fact.get("description")
                or fact.get("text")
            )
        }
        rel_entry["followup_question"] = generate_follow_up_question(rel_entry)
        relationships.append(rel_entry)

    episodes_data = _unwrap_structured(episodes_struct)
    for episode in episodes_data.get("episodes", []) or []:
        if not isinstance(episode, dict):
            continue

        metadata = episode.get("metadata") or {}
        episodes.append({
            "name": episode.get("name"),
            "content": (
                episode.get("content")
                or episode.get("episode_body")
                or episode.get("text")
            ),
            "reference_time": (
                episode.get("reference_time")
                or metadata.get("reference_time")
                or metadata.get("timestamp")
            )
        })

    return entities, relationships, episodes


def generate_follow_up_question(relationship):
    """Create a follow-up question for a single relationship."""
    if not relationship:
        return None

    fact = relationship.get("fact")
    rel_type = relationship.get("type") or "relationship"
    source = relationship.get("from") or "this entity"
    target = relationship.get("to") or "the related party"

    if fact:
        return fact
    return f"What does the {rel_type} between {source} and {target} reveal?"


def extract_query_terms(question: str):
    """Lowercase keywords from question for filtering."""
    if not question:
        return set()
    tokens = re.findall(r"\w+", question.lower())
    return {token for token in tokens if len(token) > 3}


def filter_episodes(episodes, entity_names, query_terms):
    """Prefer episodes that mention detected entities or query terms."""
    if not episodes:
        return episodes

    def episode_matches(ep):
        content = (ep.get("content") or "").lower()
        if any(name.lower() in content for name in entity_names if name):
            return True
        if any(term in content for term in query_terms):
            return True
        return False

    filtered = [ep for ep in episodes if episode_matches(ep)]
    return filtered or episodes


def build_entity_snippets(entities, doc_results, max_snippets=5):
    """Extract contextual snippets from documents for detected entities."""
    snippets = []
    if not entities or not doc_results:
        return snippets

    for entity in entities:
        name = (entity or {}).get("name")
        if not name:
            continue
        lowered = name.lower()

        for hit in doc_results:
            text = hit.payload.get("text", "")
            idx = text.lower().find(lowered)
            if idx == -1:
                continue

            start = max(0, idx - 200)
            end = min(len(text), idx + len(name) + 200)
            snippet = text[start:end].strip()

            snippets.append({
                "name": name,
                "source": hit.payload.get("title"),
                "snippet": snippet
            })
            break  # Move to next entity

        if len(snippets) >= max_snippets:
            break

    return snippets


def build_relationship_timeline(relationships, max_entries=6):
    """Create temporal bullets from relationship data."""
    entries = []
    if not relationships:
        return entries

    for rel in relationships:
        fact_text = rel.get("fact")
        if not fact_text:
            continue
        timestamp = rel.get("valid_from") or rel.get("valid_to")
        label = timestamp or "undated"
        entries.append(f"{label}: {fact_text}")
        if len(entries) >= max_entries:
            break

    return entries


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
        (
            graph_entities,
            graph_relationships,
            graph_episodes,
            graph_preview
        ) = await query_graphiti(request.question)
        entity_names = [entity.get("name") for entity in graph_entities if entity.get("name")]
        query_terms = extract_query_terms(request.question)

        # 2. Query Qdrant (document chunks)
        print(f"Querying Qdrant: {request.question}")
        doc_results = query_qdrant(request.question, permissions)

        graph_episodes = filter_episodes(graph_episodes, entity_names, query_terms)

        # 3. Build combined context
        context_parts = []

        # Add graph context
        context_parts.append("KNOWLEDGE GRAPH INSIGHTS:")
        if graph_entities:
            entity_preview = ", ".join(
                filter(None, [entity.get("name") for entity in graph_entities[:8]])
            )
            context_parts.append(f"Entities: {entity_preview}")
        if graph_relationships:
            rel_preview = "; ".join(
                filter(
                    None,
                    [
                        f"{rel.get('from')} -[{rel.get('type')}]-> {rel.get('to')}"
                        for rel in graph_relationships[:5]
                    ]
                )
            )
            context_parts.append(f"Relationships: {rel_preview}")
            for rel in graph_relationships[:5]:
                fact_text = rel.get("fact")
                if fact_text:
                    context_parts.append(f"Relationship detail: {fact_text}")
        if graph_episodes:
            episode_preview = "; ".join(
                filter(
                    None,
                    [
                        f"{episode.get('name')} ({episode.get('reference_time')})"
                        for episode in graph_episodes[:3]
                    ]
                )
            )
            context_parts.append(f"Episodes: {episode_preview}")
        if not any([graph_entities, graph_relationships, graph_episodes]) and graph_preview:
            context_parts.append(graph_preview[:500])
        context_parts.append("")

        # Add document context
        context_parts.append("RELEVANT DOCUMENTS:")
        for i, hit in enumerate(doc_results, 1):
            context_parts.append(f"\n[Document {i}: {hit.payload['title']}]")
            context_parts.append(f"{hit.payload['text'][:800]}")  # First 800 chars

        # Entity-specific snippets
        entity_snippets = build_entity_snippets(graph_entities, doc_results)
        if entity_snippets:
            context_parts.append("\nENTITY PROFILES:")
            for profile in entity_snippets:
                context_parts.append(
                    f"- {profile['name']} ({profile['source']}): {profile['snippet']}"
                )

        # Timeline bullets from relationships
        timeline_entries = build_relationship_timeline(graph_relationships)
        if timeline_entries:
            context_parts.append("\nTIMELINE HIGHLIGHTS:")
            for entry in timeline_entries:
                context_parts.append(f"- {entry}")

        # Fallback episodes from Qdrant hits if Graphiti has none yet
        if doc_results:
            fallback_episodes = []
            for hit in doc_results:
                mentions = set(hit.payload.get("mentions", []))
                hit_text = hit.payload.get("text", "")
                mention_match = any(
                    name and name.lower() in hit_text.lower()
                    for name in entity_names
                ) or mentions.intersection(entity_names)
                term_match = any(term in hit_text.lower() for term in query_terms)

                if entity_names and not (mention_match or term_match):
                    continue

                fallback_episodes.append({
                    "name": hit.payload.get("title"),
                    "content": hit.payload.get("text", "")[:1200],
                    "reference_time": hit.payload.get("reference_time")
                })

            if not graph_episodes:
                graph_episodes = fallback_episodes or [
                    {
                        "name": hit.payload.get("title"),
                        "content": hit.payload.get("text", "")[:1200],
                        "reference_time": hit.payload.get("reference_time")
                    }
                    for hit in doc_results[:2]
                ]
            else:
                existing_names = {ep.get("name") for ep in graph_episodes}
                for episode in fallback_episodes:
                    if episode.get("name") not in existing_names:
                        graph_episodes.append(episode)

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
            "graph_context": (graph_preview or "")[:200],  # Preview
            "graph_entities": graph_entities,
            "graph_relationships": graph_relationships,
            "graph_episodes": filter_episodes(graph_episodes, entity_names, query_terms)
        }

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "graph_context": "",
            "graph_entities": [],
            "graph_relationships": [],
            "graph_episodes": []
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