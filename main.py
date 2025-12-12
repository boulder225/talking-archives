# main.py - FastAPI backend
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
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
import uuid
import aiofiles
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

load_dotenv()

app = FastAPI()

# Directories
DOCUMENTS_DIR = Path("documents")
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

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
    """Extract potential entity names from question text."""
    if not question:
        return []
    
    # First, try to extract full proper names (handles consecutive capitals like "MacGuigan")
    # Pattern: Capital letter, then lowercase, optionally followed by capital+lowercase (for names like MacGuigan)
    # Then optionally space + capital+lowercase (for first+last names)
    full_name_pattern = r"[A-Z][a-z]+(?:[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+(?:[A-Z][a-z]+)*)*"
    full_matches = re.findall(full_name_pattern, question)
    
    # Also get simple capitalized words/phrases
    simple_pattern = r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
    simple_matches = re.findall(simple_pattern, question)
    
    # Prefer full matches (handle consecutive capitals), filter out fragments
    stopwords = {"Who", "What", "Where", "When", "Why", "How", "Summarize", "Tell", "Show", "Write", "Wrote"}
    guesses = []
    
    # First, prioritize full matches (they handle names with consecutive capitals)
    for match in full_matches:
        match = match.strip()
        if not match or match in stopwords:
            continue
        
        # Filter out very short or very long matches
        if len(match) < 4 or len(match.split()) > 4:
            continue
        
        # Check if this is a fragment/subset of another full match
        is_fragment = any(
            match != other and (match in other or match.lower() in other.lower())
            for other in full_matches
            if other != match
        )
        if is_fragment:
            continue
            
        guesses.append(match)
    
    # Only add simple matches that aren't already covered by full matches
    for match in simple_matches:
        match = match.strip()
        if not match or match in stopwords:
            continue
        
        # Filter out very short or very long matches
        if len(match) < 4 or len(match.split()) > 4:
            continue
        
        # Skip if it's already in full matches or is a fragment
        is_already_covered = any(
            match in full_match or match.lower() in full_match.lower()
            for full_match in full_matches
        )
        if is_already_covered:
            continue
        
        # Check if it's a subset of another match
        is_subset = any(
            match != other and (match in other or match.lower() in other.lower())
            for other in full_matches + simple_matches
            if other != match
        )
        if is_subset:
            continue
            
        guesses.append(match)
    
    # Sort by length (longest first) to prioritize full names
    guesses.sort(key=len, reverse=True)
    
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


def generate_contextual_followup(entity_name: str, context: str) -> str:
    """Generate a meaningful follow-up question based on the relationship context."""
    # Clean up entity name for better questions
    if not entity_name or entity_name.strip() == "":
        entity_name = "this entity"
    else:
        # If entity name is too long (like addresses), shorten it
        if len(entity_name) > 50:
            # Try to extract the main part (before comma or first few words)
            if ',' in entity_name:
                entity_name = entity_name.split(',')[0].strip()
            else:
                words = entity_name.split()
                if len(words) > 6:
                    entity_name = ' '.join(words[:6]) + "..."
    
    context_lower = context.lower()
    
    # Analyze the context to determine what kind of question would be most relevant
    if any(word in context_lower for word in ["published", "mailed", "distributed", "scattered", "received"]):
        return f"How was {entity_name} distributed and what was its impact?"
    
    elif any(word in context_lower for word in ["committee", "organization", "group", "association", "party"]):
        return f"What role did {entity_name} play in these events?"
    
    elif any(word in context_lower for word in ["report", "study", "investigation", "findings"]):
        return f"What were the key findings about {entity_name}?"
    
    elif any(word in context_lower for word in ["date", "year", "month", "time", "when"]) and any(char.isdigit() for char in context):
        return f"What happened with {entity_name} during this time period?"
    
    elif any(word in context_lower for word in ["location", "address", "street", "city", "province", "from", "to"]):
        return f"What is the significance of {entity_name} in this location?"
    
    elif any(word in context_lower for word in ["content", "message", "text", "entitled", "titled"]):
        return f"What was the content and purpose of {entity_name}?"
    
    elif any(word in context_lower for word in ["effect", "impact", "result", "consequence", "outcome"]):
        return f"What were the effects and consequences of {entity_name}?"
    
    elif any(word in context_lower for word in ["law", "legal", "court", "justice", "regulation"]):
        return f"What legal implications were associated with {entity_name}?"
    
    else:
        # Default to a more specific question based on context length and content
        if len(context) > 100:
            return f"What was the broader context and significance of {entity_name}?"
        else:
            return f"What more details are available about {entity_name}?"

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
            temperature=0.0,
            max_tokens=10,
            seed=42
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
    excluded_sources: Optional[List[str]] = None

def get_all_group_ids():
    """Get all available group_ids by converting document IDs to sanitized group_ids."""
    doc_ids = list_document_ids()
    # Convert document IDs to sanitized group_ids (same logic as ingestion)
    import re
    group_ids = [re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id) for doc_id in doc_ids]
    return group_ids


def group_id_to_doc_id(group_id: str) -> Optional[str]:
    """Convert a group_id back to the original document ID."""
    if not group_id:
        return None
    doc_ids = list_document_ids()
    import re
    for doc_id in doc_ids:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
        if sanitized == group_id:
            return doc_id
    return None

async def query_graphiti(question: str, allowed_sources: Optional[List[str]] = None):
    """Query Graphiti knowledge graph for nodes, facts, and episodes."""
    # Get group_ids - filter by allowed sources if provided
    if allowed_sources is not None:
        # Convert allowed doc_ids to group_ids
        import re
        group_ids = [re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id) for doc_id in allowed_sources]
    else:
        # Get all available group_ids to search across all documents
        group_ids = get_all_group_ids()
    
    nodes_struct, nodes_preview = await call_graphiti_tool(
        "search_nodes",
        {
            "query": question,
            "group_ids": group_ids,
            "max_nodes": 10
        }
    )

    facts_struct, facts_preview = await call_graphiti_tool(
        "search_memory_facts",
        {
            "query": question,
            "group_ids": group_ids,
            "max_facts": 10
        }
    )
    
    # Debug: print what we're getting from facts search
    print(f"DEBUG: Facts search for '{question}' returned:")
    print(f"  facts_struct: {facts_struct}")
    print(f"  facts_preview: {facts_preview[:200] if facts_preview else 'None'}...")
    
    print(f"DEBUG: About to call get_episodes...")

    episodes_struct, episodes_preview = await call_graphiti_tool(
        "get_episodes",
        {
            "group_ids": group_ids,
            "max_episodes": 5
        }
    )
    
    print(f"DEBUG: About to call normalize_graphiti_results...")

    (
        graph_entities,
        graph_relationships,
        graph_episodes
    ) = normalize_graphiti_results(
        nodes_struct,
        facts_struct,
        episodes_struct
    )
    
    print(f"DEBUG: normalize_graphiti_results returned {len(graph_relationships)} relationships")

    preview_parts = list(
        filter(
            None,
            [nodes_preview, facts_preview, episodes_preview]
        )
    )
    preview_text = "\n\n".join(preview_parts)

    return graph_entities, graph_relationships, graph_episodes, preview_text, facts_struct


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

        async with httpx.AsyncClient(timeout=45.0) as client:
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
    print(f"DEBUG: normalize_graphiti_results called with facts_struct: {type(facts_struct)}")
    entities = []
    relationships = []
    episodes = []

    # Build episode group_id mapping for entity attribution
    episodes_data = _unwrap_structured(episodes_struct)
    episode_to_group_id = {}
    for episode in episodes_data.get("episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        episode_name = episode.get("name", "")
        metadata = episode.get("metadata") or {}
        # Try to extract group_id from episode name (format: "DOC_NAME (Part X)")
        if episode_name:
            # Extract base document name before " (Part"
            base_name = episode_name.split(" (Part")[0]
            # Convert to group_id format (sanitized)
            import re
            group_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
            episode_to_group_id[episode_name] = group_id
        # Also check metadata
        if metadata.get("group_id"):
            episode_to_group_id[episode_name] = metadata.get("group_id")

    nodes_data = _unwrap_structured(nodes_struct)
    for node in nodes_data.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        labels = node.get("labels") or []
        attributes = node.get("attributes") or {}
        entity_type = labels[0] if labels else attributes.get("type")
        
        # Try to extract group_id from node attributes or metadata
        group_id = attributes.get("group_id") or node.get("group_id")
        
        entity = {
            "name": node.get("name") or node.get("label"),
            "type": entity_type,
            "uuid": node.get("uuid")
        }
        
        # Add group_id if we found it
        if group_id:
            entity["group_id"] = group_id
        
        entities.append(entity)

    facts_data = _unwrap_structured(facts_struct)
    
    for fact in facts_data.get("facts", []) or []:
        if not isinstance(fact, dict):
            continue

        # Extract the fact text which contains the relationship information
        fact_text = fact.get("fact", "")
        relationship_name = fact.get("name", "")
        
        # Generate a better follow-up question based on the actual fact content
        if fact_text:
            followup_question = generate_fact_based_followup(fact_text, relationship_name)
            
            rel_entry = {
                "from": "",  # We don't have direct node names, the fact text contains the full relationship
                "to": "",
                "type": relationship_name or "relationship",
                "valid_from": fact.get("valid_at"),
                "valid_to": fact.get("expired_at"),
                "fact": fact_text,
                "followup_question": followup_question
            }
            relationships.append(rel_entry)

    # Process episodes (already loaded above)
    for episode in episodes_data.get("episodes", []) or []:
        if not isinstance(episode, dict):
            continue

        metadata = episode.get("metadata") or {}
        episode_name = episode.get("name", "")
        
        # Extract group_id from episode name or metadata
        group_id = None
        if episode_name in episode_to_group_id:
            group_id = episode_to_group_id[episode_name]
        elif metadata.get("group_id"):
            group_id = metadata.get("group_id")
        
        episode_entry = {
            "name": episode_name,
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
        }
        
        if group_id:
            episode_entry["group_id"] = group_id
        
        episodes.append(episode_entry)
        
        # Map entities to group_id based on episode content
        # If entity appears in episode, attribute it to this document
        if group_id and episode_entry.get("content"):
            episode_content = episode_entry["content"].lower()
            for entity in entities:
                entity_name = entity.get("name", "").lower()
                if entity_name and entity_name in episode_content and not entity.get("group_id"):
                    entity["group_id"] = group_id

    return entities, relationships, episodes


def generate_fact_based_followup(fact_text: str, relationship_type: str = "") -> str:
    """Generate a relevant follow-up question based on the actual fact content."""
    if not fact_text:
        return "What more information is available about this topic?"
    
    fact_lower = fact_text.lower()
    
    # Extract the main subject from the fact (usually the first entity mentioned)
    import re
    
    # Try to extract person names (capitalized words before "was" or "is")
    person_match = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(?:was|is)', fact_text)
    if person_match:
        person_name = person_match.group(1)
        
        # Generate questions based on relationship type and content
        if any(word in fact_lower for word in ["principal", "dean", "professor", "associate professor"]):
            return f"What other roles did {person_name} have in academia?"
        elif any(word in fact_lower for word in ["committee", "member", "chairman"]):
            return f"What was {person_name}'s contribution to this committee?"
        elif any(word in fact_lower for word in ["received", "mailed", "distributed"]):
            return f"What was the impact of materials involving {person_name}?"
        else:
            return f"What other activities was {person_name} involved in?"
    
    # Handle organizational or location-based facts
    if any(word in fact_lower for word in ["university", "college", "faculty"]):
        institution_match = re.search(r'((?:[A-Z][a-z]+\s+)*University|(?:[A-Z][a-z]+\s+)*College)', fact_text)
        if institution_match:
            institution = institution_match.group(1)
            return f"What role did {institution} play in these events?"
    
    # Handle pamphlet/document distribution
    if any(word in fact_lower for word in ["pamphlet", "leaflet", "distributed", "scattered", "mailed"]):
        if "students" in fact_lower:
            return "How did students respond to these materials?"
        elif "campus" in fact_lower:
            return "What was the campus reaction to these materials?"
        else:
            return "What was the broader impact of this distribution?"
    
    # Handle geographic/location facts
    if any(word in fact_lower for word in ["montreal", "toronto", "ontario", "quebec", "hamilton"]):
        location_match = re.search(r'([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)', fact_text)
        if location_match and any(word in location_match.group(1).lower() for word in ["montreal", "toronto", "ontario", "quebec", "hamilton"]):
            return f"What other activities occurred in this region during this period?"
    
    # Default based on relationship type
    if relationship_type:
        if "AFFILIATED_WITH" in relationship_type or "MEMBER" in relationship_type:
            return "What other institutional connections existed?"
        elif "DISTRIBUTED" in relationship_type or "RECEIVED" in relationship_type:
            return "How widespread was this distribution network?"
        elif "COMPARED_WITH" in relationship_type:
            return "What were the key differences in approaches?"
    
    # Generic fallback
    return "What additional context is available about this topic?"

def generate_follow_up_question(relationship):
    """Create a follow-up question for a single relationship."""
    if not relationship:
        return None

    fact = relationship.get("fact", "")
    target = relationship.get("to") or relationship.get("from") or "this entity"
    
    # Use the contextual followup generator for better questions
    return generate_contextual_followup(target, fact)


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


def verify_entity_in_document(entity_name: str, doc_id: str) -> bool:
    """Verify that an entity actually exists in the specified document."""
    try:
        from ingest import read_document_text
        text = read_document_text(doc_id)
        # Case-insensitive search
        return entity_name.lower() in text.lower()
    except Exception as e:
        print(f"Warning: Could not verify entity '{entity_name}' in '{doc_id}': {e}")
        return False


def map_entity_to_documents(entity_name: str) -> List[str]:
    """Find which documents actually contain this entity by searching document text."""
    matching_docs = []
    doc_ids = list_document_ids()
    
    for doc_id in doc_ids:
        if verify_entity_in_document(entity_name, doc_id):
            matching_docs.append(doc_id)
    
    return matching_docs


def query_qdrant(question: str, user_permissions: list, excluded_sources: Optional[List[str]] = None):
    """Query Qdrant vector database with hybrid search (semantic + keyword)"""
    # Embed question
    embedding = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Get active sources
    active_sources = get_active_sources()
    excluded_sources = excluded_sources or []
    # Filter out excluded sources
    allowed_sources = [s for s in active_sources if s not in excluded_sources]
    
    if not allowed_sources:
        return []

    # Search with governance filter and source filter
    filter_conditions = [
        FieldCondition(
            key="access_level",
            match=MatchAny(any=user_permissions)
        )
    ]
    
    # Add source filter (filter by payload.doc_id)
    if allowed_sources:
        filter_conditions.append(
            FieldCondition(
                key="doc_id",
                match=MatchAny(any=allowed_sources)
            )
        )
    
    # Primary semantic search
    results = qdrant.query_points(
        collection_name="archive_documents",
        query=embedding,
        query_filter=Filter(must=filter_conditions),
        limit=15,
        with_payload=True
    )

    semantic_results = results.points
    
    # Extract key terms for keyword boost
    import re
    key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
    key_terms.extend(re.findall(r'"([^"]+)"', question))  # Quoted terms
    
    # Check if query looks like an entity name (2-3 capitalized words)
    is_likely_entity_query = len(key_terms) >= 2 and len(key_terms) <= 3 and all(len(term.split()) <= 2 for term in key_terms)
    
    # If we have specific terms, do additional keyword search
    if key_terms:
        # Get more results for keyword filtering
        extended_results = qdrant.query_points(
            collection_name="archive_documents",
            query=embedding,
            query_filter=Filter(must=filter_conditions),
            limit=50,  # Get more for keyword filtering
            with_payload=True
        )
        
        # Boost results that contain key terms
        boosted_results = []
        seen_ids = set()
        keyword_match_results = []
        
        # First, prioritize results that contain ALL key terms (exact match for entity names)
        if is_likely_entity_query:
            full_entity = " ".join(key_terms).lower()
            for hit in extended_results.points:
                text = hit.payload.get('text', '').lower()
                # Check for exact entity name match (case-insensitive)
                if full_entity in text:
                    if hit.id not in seen_ids:
                        hit.score = hit.score * 5.0  # Very strong boost for exact matches
                        keyword_match_results.append(hit)
                        seen_ids.add(hit.id)
        
        # Then add results that contain any key term
        for hit in extended_results.points:
            text = hit.payload.get('text', '').lower()
            if any(term.lower() in text for term in key_terms):
                if hit.id not in seen_ids:
                    # Strong boost for keyword matches
                    boost_factor = 2.0
                    # Extra boost for longer, more descriptive content
                    if len(text) > 1000 and any(len(term) > 5 for term in key_terms):
                        boost_factor = 3.0
                    hit.score = hit.score * boost_factor
                    boosted_results.append(hit)
                    seen_ids.add(hit.id)
        
        # Combine: exact matches first, then keyword matches, then semantic
        final_results = keyword_match_results + boosted_results
        
        # Add semantic results that weren't already included
        for hit in semantic_results:
            if hit.id not in seen_ids:
                final_results.append(hit)
                seen_ids.add(hit.id)
        
        # Sort by boosted scores and limit
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Normalize scores to cap at 1.0 (100%) while preserving relative ordering
        if final_results:
            max_score = max(hit.score for hit in final_results)
            if max_score > 1.0:
                # Normalize: scale all scores so the max is 1.0
                for hit in final_results:
                    hit.score = min(hit.score / max_score, 1.0)
        
        return final_results[:15]
    
    return semantic_results

@app.post("/query")
async def handle_query(request: QueryRequest):
    """Main query endpoint"""
    try:
        # User permissions (hardcoded for PoC)
        permissions = ["public", "researcher"]

        # Get excluded sources (temporary exclusion)
        excluded_sources = request.excluded_sources or []
        excluded_set = set(excluded_sources)
        
        # Calculate allowed sources (active sources minus excluded)
        active_sources = get_active_sources()
        allowed_sources = [s for s in active_sources if s not in excluded_set]
        
        # Convert excluded doc_ids to group_ids for filtering
        import re
        excluded_group_ids = set([re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id) for doc_id in excluded_sources])

        # 1. Query Graphiti (knowledge graph) - only query allowed sources
        print(f"Querying Graphiti: {request.question} (allowed sources: {allowed_sources})")
        (
            graph_entities,
            graph_relationships,
            graph_episodes,
            graph_preview,
            facts_struct
        ) = await query_graphiti(request.question, allowed_sources=allowed_sources)
        graph_entities = graph_entities or []
        
        # Filter Graphiti results by excluded sources (using group_id comparison)
        if excluded_group_ids:
            graph_entities = [
                entity for entity in graph_entities
                if entity.get("group_id") not in excluded_group_ids
            ]
            graph_relationships = [
                rel for rel in (graph_relationships or [])
                if rel.get("group_id") not in excluded_group_ids
            ]
            graph_episodes = [
                ep for ep in (graph_episodes or [])
                if ep.get("group_id") not in excluded_group_ids
            ]
        
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
        
        # Filter guessed entities: ensure they can only be mapped to allowed sources
        # This will be handled during verification, but we pre-mark them to avoid excluded sources

        # 2. Query Qdrant (document chunks)
        print(f"Querying Qdrant: {request.question}")
        doc_results = query_qdrant(request.question, permissions, excluded_sources)
        
        # Filter Qdrant results by excluded sources
        if excluded_set:
            doc_results = [
                hit for hit in doc_results
                if hit.payload.get("doc_id") not in excluded_set
            ]

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
        
        # Verify and map entities to their actual documents
        for entity in graph_entities:
            entity_name = entity.get("name")
            if not entity_name:
                continue
            
            # If entity already has group_id, verify it's correct and not excluded
            current_group_id = entity.get("group_id")
            if current_group_id:
                # First check if this group_id is from an excluded source
                if current_group_id in excluded_group_ids:
                    print(f"⚠️  Entity '{entity_name}' from excluded source (group_id: {current_group_id}) - marking for removal")
                    entity["_not_found_in_docs"] = True
                    continue  # Skip further verification for excluded entities
                
                # Convert group_id back to doc_id format for verification
                # Try to find matching doc_id
                doc_ids = list_document_ids()
                import re
                matching_doc = None
                for doc_id in doc_ids:
                    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
                    if sanitized == current_group_id:
                        matching_doc = doc_id
                        break
                
                # Check if matching document is excluded
                if matching_doc and matching_doc in excluded_set:
                    print(f"⚠️  Entity '{entity_name}' from excluded document '{matching_doc}' - marking for removal")
                    entity["_not_found_in_docs"] = True
                    continue  # Skip further verification for excluded entities
                
                # Verify entity exists in that document
                if matching_doc:
                    if not verify_entity_in_document(entity_name, matching_doc):
                        # Entity not in attributed document - clear group_id and find correct one
                        print(f"⚠️  Entity '{entity_name}' attributed to '{matching_doc}' but not found there. Searching for correct document...")
                        entity["group_id"] = None
                        entity["_verification_failed"] = matching_doc
            
            # If entity doesn't have group_id or verification failed, find correct documents
            if not entity.get("group_id"):
                matching_docs = map_entity_to_documents(entity_name)
                # Filter matching docs to only include allowed sources
                matching_docs = [doc for doc in matching_docs if doc not in excluded_set]
                if matching_docs:
                    # Use first matching document
                    matching_doc = matching_docs[0]
                    import re
                    entity["group_id"] = re.sub(r'[^a-zA-Z0-9_-]', '_', matching_doc)
                    entity["doc_id"] = matching_doc  # Store original doc_id too
                    print(f"✓ Mapped entity '{entity_name}' to document '{matching_doc}'")
                else:
                    # Entity not found in allowed documents - mark for removal
                    print(f"⚠️  Entity '{entity_name}' not found in allowed documents")
                    entity["_not_found_in_docs"] = True

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

        # If we have Graphiti facts, replace any broken relationships with better ones
        if facts_struct:
            # Check if relationships have empty 'to' fields (indicating broken normalization)
            has_meaningful_relationships = graph_relationships and any(rel.get("to") for rel in graph_relationships)
            if not has_meaningful_relationships:
                # Clear broken relationships and create new ones from facts
                graph_relationships = []
                facts_data = _unwrap_structured(facts_struct)
                seen_questions = set()
                
                for fact in facts_data.get("facts", [])[:10]:  # Limit to 10 facts
                    if not isinstance(fact, dict):
                        continue
                        
                    fact_text = fact.get("fact", "")
                    if not fact_text:
                        continue
                        
                    # Generate a better follow-up question
                    followup = generate_fact_based_followup(fact_text, fact.get("name", ""))
                    
                    # Only add unique questions
                    if followup not in seen_questions:
                        seen_questions.add(followup)
                        graph_relationships.append({
                            "from": "",
                            "to": "",
                            "type": fact.get("name", "relationship"),
                            "fact": fact_text[:200],
                            "followup_question": followup
                        })
        
        # Fallback: Create relationships from Graphiti entities if still no relationships
        elif not graph_relationships and entity_snippets:
            seen_questions = set()  # Track unique follow-up questions
            
            for snippet in entity_snippets:
                # Create clean, readable relationship facts
                entity_name = snippet["name"]
                source_doc = snippet["source"]
                raw_snippet = snippet["snippet"]
                
                # Generate a clean fact statement
                clean_fact = f"{source_doc} mentions {entity_name}"
                
                # Try to extract key context from snippet
                # Look for the sentence containing the entity name
                sentences = raw_snippet.split('. ')
                best_sentence = ""
                
                for sentence in sentences:
                    if entity_name.lower() in sentence.lower() and len(sentence) < 200:
                        best_sentence = sentence.strip()
                        if not best_sentence.endswith('.'):
                            best_sentence += '.'
                        break
                
                if best_sentence:
                    clean_fact = f"{source_doc} mentions {entity_name}: {best_sentence}"
                elif sentences and len(sentences[0]) < 150:
                    context = sentences[0].strip()
                    if context and not context.endswith('.'):
                        context += '.'
                    clean_fact = f"{source_doc} mentions {entity_name}: {context}"
                
                # Generate a meaningful follow-up question based on the relationship context
                followup = generate_contextual_followup(entity_name, best_sentence or raw_snippet)
                
                # Only add if we haven't seen this follow-up question before
                if followup not in seen_questions:
                    seen_questions.add(followup)
                    graph_relationships.append({
                        "from": source_doc,
                        "to": entity_name,
                        "type": "mentions",
                        "fact": clean_fact[:200],  # Limit length
                        "followup_question": followup
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
5. Be honest about uncertainty
6. Provide consistent, deterministic responses"""
                },
                {
                    "role": "user",
                    "content": f"""CONTEXT:
{context}

QUESTION: {request.question}

Provide a comprehensive answer with citations to document titles. If the context doesn't contain relevant information, say so."""
                }
            ],
            temperature=0.0,
            max_tokens=800,
            seed=42,
            top_p=1.0
        )

        answer = response.choices[0].message.content

        # Final filter to remove any noise entities before returning
        graph_entities = [
            entity for entity in graph_entities
            if not is_noise_entity(entity.get("name", ""))
        ]
        
        # Final filter: Remove entities from excluded sources (check both doc_id and group_id)
        if excluded_set or excluded_group_ids:
            final_filtered_entities = []
            for entity in graph_entities:
                # Check doc_id first
                entity_doc_id = entity.get("doc_id")
                if entity_doc_id and entity_doc_id in excluded_set:
                    continue  # Skip entities from excluded documents
                
                # Check group_id as backup
                entity_group_id = entity.get("group_id")
                if entity_group_id and entity_group_id in excluded_group_ids:
                    continue  # Skip entities from excluded group_ids
                
                # Also skip entities marked as not found in allowed docs
                if entity.get("_not_found_in_docs"):
                    continue
                
                final_filtered_entities.append(entity)
            graph_entities = final_filtered_entities
        
        # Ensure all entities have doc_id for frontend display
        # But also filter out any that map to excluded sources
        final_entities = []
        for entity in graph_entities:
            if not entity.get("doc_id") and entity.get("group_id"):
                doc_id = group_id_to_doc_id(entity.get("group_id"))
                if doc_id:
                    entity["doc_id"] = doc_id
                    # Check again if this doc_id is excluded
                    if doc_id in excluded_set:
                        continue  # Skip entities from excluded documents
            
            # Final check: skip if entity is from excluded source
            if entity.get("doc_id") and entity.get("doc_id") in excluded_set:
                continue
            if entity.get("group_id") and entity.get("group_id") in excluded_group_ids:
                continue
            if entity.get("_not_found_in_docs"):
                continue
                
            final_entities.append(entity)
        graph_entities = final_entities

        # 5. Deduplicate sources and keep highest relevance score
        sources_dict = {}
        for hit in doc_results:
            title = hit.payload["title"]
            score = hit.score
            if title not in sources_dict or score > sources_dict[title]["relevance"]:
                sources_dict[title] = {
                    "title": title,
                    "relevance": score
                }
        
        # Sort by relevance (highest first)
        deduplicated_sources = sorted(sources_dict.values(), key=lambda x: x["relevance"], reverse=True)

        # 5. Return structured response
        return {
            "answer": answer,
            "sources": deduplicated_sources,
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

    errors = []
    
    # Try to delete from Qdrant
    try:
        delete_qdrant_points([doc_id])
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "errno 61" in error_str or "ConnectError" in str(type(e).__name__):
            errors.append(f"Qdrant (port 6333) not available - data not removed from vector database")
        else:
            errors.append(f"Failed to delete from Qdrant: {error_str}")
    
    # Try to delete from Graphiti
    try:
        await clear_graphiti_groups([doc_id])
    except Exception as e:
        error_str = str(e)
        if "Connection refused" in error_str or "All connection attempts failed" in error_str or "ConnectError" in str(type(e).__name__):
            errors.append(f"Graphiti (port 8000) not available - data not removed from knowledge graph")
        else:
            errors.append(f"Failed to delete from Graphiti: {error_str}")
    
    # Always mark as inactive in local state
    set_source_active(doc_id, False)
    
    if errors:
        # Return partial success with warnings
        return {
            "status": "partially_removed",
            "doc_id": doc_id,
            "message": f"Document marked as inactive locally, but some services were unavailable: {'; '.join(errors)}"
        }
    
    return {"status": "removed", "doc_id": doc_id}


@app.post("/upload-ocr")
async def upload_ocr(
    file: UploadFile = File(...),
    doc_name: Optional[str] = Form(None)
):
    """
    Upload an image (JPG, PNG, TIFF) or PDF file for OCR processing.
    Extracts text and adds it as a new document to the archive.
    """
    # Validate file type
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename for temporary storage
    temp_id = str(uuid.uuid4())
    temp_file_path = UPLOADS_DIR / f"{temp_id}{file_ext}"
    
    try:
        # Save uploaded file
        async with aiofiles.open(temp_file_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)
        
        # Extract text using OCR
        extracted_text = None
        
        if file_ext == '.pdf':
            # Convert PDF to images and extract text from each page
            try:
                images = convert_from_path(str(temp_file_path))
                text_parts = []
                for img in images:
                    text = pytesseract.image_to_string(img)
                    text_parts.append(text)
                extracted_text = '\n\n'.join(text_parts)
            except Exception as e:
                error_str = str(e)
                if "Connection refused" in error_str or "errno 61" in error_str:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Error processing PDF (connection issue): {error_str}. Make sure poppler is installed (macOS: brew install poppler)."
                    )
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing PDF: {error_str}. Make sure poppler is installed."
                )
        else:
            # Process image directly
            try:
                img = Image.open(temp_file_path)
                extracted_text = pytesseract.image_to_string(img)
            except Exception as e:
                error_str = str(e)
                if "Connection refused" in error_str or "errno 61" in error_str:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Error processing image (connection issue): {error_str}. Check tesseract installation."
                    )
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image: {error_str}"
                )
        
        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the file. Please check the image quality."
            )
        
        # Generate document ID from filename or provided name
        if doc_name:
            doc_id = doc_name.strip()
        else:
            doc_id = Path(file.filename).stem
        
        # Ensure doc_id is safe and unique
        doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
        if not doc_id:
            doc_id = f"ocr_document_{temp_id[:8]}"
        
        # Check if document already exists, append number if needed
        new_doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        if new_doc_path.exists():
            counter = 1
            while (DOCUMENTS_DIR / f"{doc_id}_{counter}.txt").exists():
                counter += 1
            doc_id = f"{doc_id}_{counter}"
            new_doc_path = DOCUMENTS_DIR / f"{doc_id}.txt"
        
        # Save extracted text as document
        async with aiofiles.open(new_doc_path, "w", encoding='utf-8') as f:
            await f.write(extracted_text)
        
        # Ingest the new document into Qdrant and Graphiti
        try:
            await ingest_single_document(
                doc_id,
                openai_client=openai_client,
                qdrant_client=qdrant,
                recreate_collection=False
            )
            set_source_active(doc_id, True)
            ingestion_success = True
        except Exception as ingest_error:
            # Check if it's a connection error
            error_msg = str(ingest_error)
            error_type = type(ingest_error).__name__
            
            # Check for various connection error indicators
            is_connection_error = (
                "Connection refused" in error_msg or 
                "errno 61" in error_msg or 
                "ConnectionError" in error_msg or
                "ConnectError" in error_type or
                "ResponseHandlingException" in error_type
            )
            
            if is_connection_error:
                # Determine which service is likely down
                service_hint = ""
                if "6333" in error_msg or "qdrant" in error_msg.lower():
                    service_hint = "Qdrant (port 6333) is not running."
                elif "8000" in error_msg or "graphiti" in error_msg.lower():
                    service_hint = "Graphiti (port 8000) is not running."
                else:
                    service_hint = "Qdrant (port 6333) or Graphiti (port 8000) may not be running."
                
                raise HTTPException(
                    status_code=503,
                    detail=f"✓ Text extracted successfully ({len(extracted_text)} characters), but failed to ingest into archive. {service_hint} The extracted text has been saved to: {new_doc_path}. Please start the required services and try re-ingesting this document later."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"✓ Text extracted successfully ({len(extracted_text)} characters), but failed to ingest: {error_msg}. The extracted text has been saved to: {new_doc_path}."
                )
        
        # Clean up temporary file
        temp_file_path.unlink(missing_ok=True)
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "message": f"Document '{doc_id}' extracted via OCR and added to archive.",
            "extracted_text_length": len(extracted_text),
            "preview": extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper error messages)
        if temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        # Clean up on any other error
        error_str = str(e)
        if temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
        
        # Check for connection errors
        if "Connection refused" in error_str or "errno 61" in error_str or "ConnectionError" in str(type(e).__name__):
            # Determine which service might be down
            detail_msg = f"Connection error: {error_str}. "
            if "6333" in error_str or "qdrant" in error_str.lower():
                detail_msg += "Qdrant (port 6333) may not be running."
            elif "8000" in error_str or "graphiti" in error_str.lower():
                detail_msg += "Graphiti (port 8000) may not be running."
            else:
                detail_msg += "One of the required services (Qdrant port 6333 or Graphiti port 8000) may not be running."
            
            raise HTTPException(status_code=503, detail=detail_msg)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {error_str}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)