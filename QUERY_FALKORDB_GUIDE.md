# How to Query FalkorDB/Graphiti

This guide explains the different ways to query FalkorDB through Graphiti for "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND" or any other query.

## Methods to Query FalkorDB

### Method 1: Using the Web Interface (Easiest)

1. **Via the Talking Archives Search Page:**
   - Go to `http://localhost:8080` (or your server IP)
   - Enter your query: `"summarize SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"`
   - Click "Search Archives"
   - The system will automatically query FalkorDB through Graphiti

2. **Via Graphiti UI directly:**
   - Open `http://localhost:3000/graph` in your browser
   - Use the search interface in the Graphiti UI to search for entities/terms
   - The UI provides an interactive graph visualization

### Method 2: Using the Python Query Script

I've created a dedicated script for querying FalkorDB directly:

```bash
# Query with your specific phrase
python query_falkordb.py "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"

# Or query anything else
python query_falkordb.py "Mark MacGuigan"
python query_falkordb.py "Raymond Labarge"
```

**What the script does:**
- Searches for **nodes (entities)** matching your query
- Searches for **facts (relationships)** related to your query  
- Retrieves **episodes (temporal events)** from the graph
- Displays results in a readable format

### Method 3: Using the API Endpoint

You can query via the FastAPI endpoint programmatically:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "summarize SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND",
    "user_id": "researcher_1",
    "excluded_sources": []
  }'
```

This returns:
- Answer with citations
- Graph entities found
- Relationships/facts
- Episodes
- Document sources

### Method 4: Direct Graphiti MCP Tool Calls

If you need to query programmatically in Python:

```python
import asyncio
from main import call_graphiti_tool, get_all_group_ids, ensure_graphiti_session

async def query():
    await ensure_graphiti_session()
    group_ids = get_all_group_ids()
    
    # Search nodes (entities)
    nodes, preview = await call_graphiti_tool(
        "search_nodes",
        {
            "query": "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND",
            "group_ids": group_ids,
            "max_nodes": 10
        }
    )
    
    # Search facts (relationships)
    facts, preview = await call_graphiti_tool(
        "search_memory_facts",
        {
            "query": "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND",
            "group_ids": group_ids,
            "max_facts": 10
        }
    )
    
    # Get episodes (temporal events)
    episodes, preview = await call_graphiti_tool(
        "get_episodes",
        {
            "group_ids": group_ids,
            "max_episodes": 5
        }
    )
    
    return nodes, facts, episodes

# Run the query
results = asyncio.run(query())
```

## Understanding the Query Results

### Nodes (Entities)
- Represents entities extracted from documents (people, places, organizations, concepts)
- Each node has:
  - `name`: Entity name
  - `labels`: Entity type
  - `attributes`: Additional properties
  - `uuid`: Unique identifier

### Facts (Relationships)
- Represents relationships between entities
- Each fact has:
  - `fact`: The relationship text
  - `name`: Relationship type
  - `valid_at`: Start time (temporal)
  - `expired_at`: End time (temporal)

### Episodes
- Temporal events or narrative segments
- Each episode has:
  - `name`: Episode title
  - `content`: Episode text
  - `reference_time`: When it occurred

## Query Tips

1. **Use specific phrases:** "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND" is better than just "libel"

2. **Try variations:**
   - "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"
   - "sedition in England"
   - "libel offences"
   - "England criminal law"

3. **Filter by document:** Use `excluded_sources` to search only specific documents

4. **Group IDs:** Documents are converted to group_ids (e.g., "APPENDIX I" → "APPENDIX_I")

## Example: Querying for "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"

```bash
# Using the query script
cd /Users/enrico/workspace/talking-archives-poc
source venv/bin/activate
python query_falkordb.py "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"
```

This will show:
- All entities related to seditious libel in England
- Facts/relationships about this topic
- Episodes (document sections) discussing it
- Which documents contain this information

## Troubleshooting

- **No results?** Try broader queries or check if the term exists in your documents
- **Want to search specific documents?** Modify `group_ids` in the script
- **Graphiti not responding?** Check if the Graphiti MCP service is running at port 8000



