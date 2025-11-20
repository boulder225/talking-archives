# Talking Archives

Talking Archives is infrastructure for Indigenous data sovereignty. Every component keeps archival materials under community ownership, control, access, and possession (OCAP) while still enabling AI-assisted research.

## Core Rules
- Data never leaves community hardware; no opaque cloud services.
- Every query, access attempt, and output is logged for audit.
- Humans can veto, flag, or approve any AI result.
- Code stays simple and reviewable over clever optimizations.

## Stack (fixed by charter)
- Backend: FastAPI (Python) on port 8080
- Knowledge graph: Graphiti + FalkorDB (docker-graphiti-mcp-1, port 8000)
- Vector store: Qdrant (port 6333)
- LLM: OpenAI GPT-4o (temporary, replaceable)
- Frontend: Static HTML + Tailwind CSS (no build tooling)

## Daily Use
1. `python ingest.py` — load or refresh local archival documents.
2. `python main.py` — run the FastAPI service (binds to 8080).
3. `python test_connections.py` — verify Graphiti, FalkorDB, and Qdrant links.

Keep the system air-gapped when needed, monitor the audit logs, and involve the community in every deployment decision. This project exists to serve them.
