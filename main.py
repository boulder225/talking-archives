# main.py - FastAPI backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from qdrant_client.http.models import PointIdsList
import httpx
from dotenv import load_dotenv
import json
import os
import re
from pathlib import Path
from typing import List, Optional
from ingest import ingest_document as ingest_single_document, list_document_ids, qdrant_point_id

load_dotenv()

app = FastAPI()

# Source state tracking
STATE_FILE = Path("sources_state.json")

# Graphiti session tracking
graphiti_session_id = None
graphiti_session_lock = asyncio.Lock()


def load_sources_state():
    if not STATE_FILE.exists():
        return {"sources": {}}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {"sources": {}}


def save_sources_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_source_entries(doc_ids: List[str]):
    state = load_sources_state()
    changed = False
    for doc_id in doc_ids:
        if doc_id not in state["sources"]:
            state["sources"][doc_id] = {"active": True}
            changed = True
    if changed:
        save_sources_state(state)
    return state


def set_source_active(doc_id: str, active: bool):
    state = load_sources_state()
    if doc_id not in state["sources"]:
        state["sources"][doc_id] = {}
    state["sources"][doc_id]["active"] = active
    save_sources_state(state)


def get_source_status(doc_id: str) -> bool:
    state = load_sources_state()
    entry = state["sources"].get(doc_id)
    if entry is None:
        return True
    return entry.get("active", True)


def get_active_sources() -> List[str]:
    doc_ids = list_document_ids()
    state = ensure_source_entries(doc_ids)
    return [
        doc_id
        for doc_id in doc_ids
        if state["sources"].get(doc_id, {}).get("active", True)
    ]


def guess_entities_from_question(question: str) -> List[str]:
    if not question:
        return []
    matches = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}", question)
    stopwords = {"Who", "What", "Where", "When", "Why", "How"}
    guesses = []
    for match in matches:
        if match in stopwords:
            continue
        guesses.append(match)
    return guesses


def is_noise_entity(entity_name: str) -> bool:
    """Check if an entity name is likely noise/fragment (structural checks only)."""
    if not entity_name:
        return True
    
    name = entity_name.strip()
    
    # Too short
    if len(name) < 3:
        return True
    
    words = name.split()
    
    # Contains newlines (fragments)
    if "\n" in name:
        return True
    
    # Very long fragments (> 8 words likely not an entity)
    if len(words) > 8:
        return True
    
    return False


def classify_entity_type(entity_name: str, context_snippets: List[str] = None) -> str:
    """Classify an entity into Person, Organization, Place, Event, Concept, or Other."""
    if not entity_name:
        return "Entity"
    
    # Use LLM for classification
    context_text = ""
    if context_snippets:
        context_text = "\n".join(context_snippets[:3])[:500]
    
    prompt = f"""Classify the following entity name into one of these types: Person, Organization, Place, Event, Concept, Other.

Entity name: {entity_name}
Context: {context_text}

Return ONLY the type name (e.g., "Person", "Organization", "Place", "Event", "Concept", "Other")."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        classified = response.choices[0].message.content.strip()
        valid_types = ["Person", "Organization", "Place", "Event", "Concept", "Other"]
        if classified in valid_types:
            return classified
    except Exception as e:
        print(f"Error classifying entity {entity_name}: {e}")
    
    return "Entity"

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
    entity_types: Optional[List[str]] = None

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


async def clear_graphiti_groups(group_ids):
    if not group_ids:
        return
    await call_graphiti_tool(
        "clear_graph",
        {
            "group_ids": group_ids
        }
    )


def delete_qdrant_points(doc_ids):
    if not doc_ids:
        return
    point_ids = [qdrant_point_id(doc_id) for doc_id in doc_ids]
    try:
        qdrant.delete(
            collection_name="archive_documents",
            points_selector=PointIdsList(points=point_ids)
        )
    except Exception as exc:
        print(f"Warning: failed to delete Qdrant points {doc_ids}: {exc}")


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
        graph_entities = graph_entities or []
        # Filter out noise entities
        graph_entities = [
            entity for entity in graph_entities
            if not is_noise_entity(entity.get("name", ""))
        ]
        entity_names = [entity.get("name") for entity in graph_entities if entity.get("name")]
        entity_types_filter = {et.lower() for et in (request.entity_types or []) if et}
        query_terms = extract_query_terms(request.question)
        if not entity_names:
            guessed = guess_entities_from_question(request.question)
            if guessed:
                graph_entities = [
                    {"name": name, "type": "Query Guess", "uuid": None}
                    for name in guessed
                ]
                entity_names = guessed

        # 2. Query Qdrant (document chunks)
        print(f"Querying Qdrant: {request.question}")
        doc_results = query_qdrant(request.question, permissions)

        initial_entity_names = [entity.get("name") for entity in graph_entities if entity.get("name")]
        graph_episodes = filter_episodes(graph_episodes, initial_entity_names, query_terms)

        # Classify entities that don't have proper types
        context_snippets = [hit.payload.get("text", "")[:200] for hit in doc_results[:3]]
        for entity in graph_entities:
            current_type = entity.get("type", "").strip()
            if not current_type or current_type == "Entity" or current_type == "Mention":
                entity_name = entity.get("name", "")
                if entity_name and not is_noise_entity(entity_name):
                    classified_type = classify_entity_type(entity_name, context_snippets)
                    entity["type"] = classified_type
        
        # Filter out any noise entities that slipped through
        graph_entities = [
            entity for entity in graph_entities
            if not is_noise_entity(entity.get("name", ""))
        ]

        # Fallback entities if none returned
        if not graph_entities and doc_results:
            mention_counts = {}
            for hit in doc_results:
                for mention in hit.payload.get("mentions", []):
                    mention_counts[mention] = mention_counts.get(mention, 0) + 1
            fallback_entities = sorted(
                mention_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )[:6]
            graph_entities = []
            for name, _ in fallback_entities:
                if not is_noise_entity(name):
                    classified_type = classify_entity_type(name, context_snippets)
                    graph_entities.append({
                        "name": name,
                        "type": classified_type,
                        "uuid": None
                    })

        entity_types_available = sorted(
            {
                entity.get("type")
                for entity in graph_entities
                if entity.get("type")
            }
        )

        if entity_types_filter:
            graph_entities = [
                entity
                for entity in graph_entities
                if (entity.get("type") or "").lower() in entity_types_filter
            ]

        entity_names = [entity.get("name") for entity in graph_entities if entity.get("name")]
        entity_snippets = build_entity_snippets(graph_entities, doc_results)

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
        if entity_snippets:
            context_parts.append("\nENTITY PROFILES:")
            for profile in entity_snippets:
                context_parts.append(
                    f"- {profile['name']} ({profile['source']}): {profile['snippet']}"
                )
        elif doc_results:
            # Build a minimal entity list from document mentions when Graphiti has none
            mention_counts = {}
            for hit in doc_results:
                for mention in hit.payload.get("mentions", []):
                    mention_counts[mention] = mention_counts.get(mention, 0) + 1
            fallback_entities = sorted(
                mention_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )[:6]
            graph_entities = [
                {"name": name, "type": "Mention", "uuid": None}
                for name, _ in fallback_entities
            ]
            entity_names.extend([name for name, _ in fallback_entities if name not in entity_names])
            entity_snippets = build_entity_snippets(graph_entities, doc_results)

        # Timeline bullets from relationships
        timeline_entries = build_relationship_timeline(graph_relationships)
        if timeline_entries:
            context_parts.append("\nTIMELINE HIGHLIGHTS:")
            for entry in timeline_entries:
                context_parts.append(f"- {entry}")

        if not graph_relationships and entity_snippets:
            for snippet in entity_snippets:
                graph_relationships.append({
                    "from": snippet["source"],
                    "to": snippet["name"],
                    "type": "mentions",
                    "fact": snippet["snippet"],
                    "followup_question": snippet["snippet"]
                })

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

        # Final filter to remove any noise entities before returning
        graph_entities = [
            entity for entity in graph_entities
            if not is_noise_entity(entity.get("name", ""))
        ]

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
            "graph_episodes": filter_episodes(graph_episodes, entity_names, query_terms),
            "available_entity_types": entity_types_available
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


@app.get("/sources")
def list_sources():
    doc_ids = list_document_ids()
    state = ensure_source_entries(doc_ids)
    sources = [
        {
            "id": doc_id,
            "active": state["sources"].get(doc_id, {}).get("active", True)
        }
        for doc_id in doc_ids
    ]
    return {"sources": sources}


@app.post("/sources/{doc_id}/reingest")
async def reingest_source(doc_id: str):
    available = list_document_ids()
    if doc_id not in available:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_qdrant_points([doc_id])
    await clear_graphiti_groups([doc_id])
    await ingest_single_document(
        doc_id,
        openai_client=openai_client,
        qdrant_client=qdrant,
        recreate_collection=False
    )
    set_source_active(doc_id, True)

    return {"status": "ok", "doc_id": doc_id}


@app.delete("/sources/{doc_id}")
async def remove_source(doc_id: str):
    available = list_document_ids()
    if doc_id not in available:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_qdrant_points([doc_id])
    await clear_graphiti_groups([doc_id])
    set_source_active(doc_id, False)

    return {"status": "removed", "doc_id": doc_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)