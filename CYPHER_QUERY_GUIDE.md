# FalkorDB Cypher Query Guide

This guide shows how to adapt the default FalkorDB query to find specific entities and relationships.

## Default Query

```cypher
MATCH (n) OPTIONAL MATCH (n)-[e]-(m) RETURN * LIMIT 100
```

This returns all nodes and their relationships (up to 100 results).

## Adapted Queries for "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"

### 1. Search Nodes by Name (Text Contains)

```cypher
MATCH (n) 
WHERE n.name CONTAINS 'SEDITIOUS' 
   OR n.name CONTAINS 'LIBEL' 
   OR n.name CONTAINS 'ENGLAND'
OPTIONAL MATCH (n)-[e]-(m) 
RETURN n, e, m 
LIMIT 100
```

### 2. Case-Insensitive Search

```cypher
MATCH (n) 
WHERE toLower(n.name) CONTAINS 'seditious' 
   OR toLower(n.name) CONTAINS 'libel'
OPTIONAL MATCH (n)-[e]-(m) 
RETURN n, e, m 
LIMIT 100
```

### 3. Search by Label/Type

```cypher
MATCH (n) 
WHERE n.name CONTAINS 'SEDITIOUS LIBEL' 
   OR n.name CONTAINS 'RELATED OFFENCES'
OPTIONAL MATCH (n)-[e]-(m) 
RETURN n, e, m, labels(n) as nodeTypes
LIMIT 100
```

### 4. Search with Properties/Attributes

```cypher
MATCH (n) 
WHERE n.name CONTAINS 'ENGLAND'
   OR (n.attributes IS NOT NULL AND 
       any(key IN keys(n.attributes) WHERE 
           n.attributes[key] CONTAINS 'SEDITIOUS' OR
           n.attributes[key] CONTAINS 'LIBEL'))
OPTIONAL MATCH (n)-[e]-(m) 
RETURN n, e, m 
LIMIT 100
```

### 5. Find Connected Entities (2 Hops)

```cypher
MATCH (n)-[e1]-(m)-[e2]-(o)
WHERE n.name CONTAINS 'SEDITIOUS' 
   OR n.name CONTAINS 'LIBEL'
RETURN n, e1, m, e2, o
LIMIT 100
```

### 6. Search Relationships/Facts

```cypher
MATCH (n)-[e]-(m)
WHERE e.fact CONTAINS 'SEDITIOUS LIBEL'
   OR e.fact CONTAINS 'RELATED OFFENCES'
   OR e.fact CONTAINS 'ENGLAND'
RETURN n, e, m, type(e) as relationshipType
LIMIT 100
```

### 7. Filter by Specific Terms (AND logic)

```cypher
MATCH (n)
WHERE n.name CONTAINS 'SEDITIOUS' 
  AND n.name CONTAINS 'LIBEL'
OPTIONAL MATCH (n)-[e]-(m)
RETURN n, e, m
LIMIT 100
```

### 8. Search All Related to a Concept

```cypher
MATCH path = (n)-[*1..2]-(m)
WHERE n.name CONTAINS 'SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND'
RETURN path
LIMIT 100
```

## Using Cypher Queries in Code

### Option 1: Via Graphiti MCP (if graph_query tool is available)

```python
import asyncio
from main import call_graphiti_tool, ensure_graphiti_session

async def query_with_cypher():
    await ensure_graphiti_session()
    
    cypher_query = """
    MATCH (n) 
    WHERE n.name CONTAINS 'SEDITIOUS' 
       OR n.name CONTAINS 'LIBEL'
    OPTIONAL MATCH (n)-[e]-(m) 
    RETURN n, e, m 
    LIMIT 100
    """
    
    result, preview = await call_graphiti_tool(
        "graph_query",  # Tool name - may vary
        {
            "query": cypher_query
        }
    )
    
    return result

# Run query
results = asyncio.run(query_with_cypher())
```

### Option 2: Direct FalkorDB Connection

If you have direct access to FalkorDB:

```python
from falkordb import Graph

# Connect to FalkorDB
graph = Graph('localhost', 6379, 'graph_name')

# Execute Cypher query
query = """
MATCH (n) 
WHERE n.name CONTAINS 'SEDITIOUS' 
   OR n.name CONTAINS 'LIBEL'
OPTIONAL MATCH (n)-[e]-(m) 
RETURN n, e, m 
LIMIT 100
"""

results = graph.query(query)
for record in results.result_set:
    print(record)
```

## Query Patterns Reference

### Basic Node Filtering

```cypher
// Exact match
WHERE n.name = 'SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND'

// Contains (substring)
WHERE n.name CONTAINS 'LIBEL'

// Case-insensitive
WHERE toLower(n.name) CONTAINS 'libel'

// Starts with
WHERE n.name STARTS WITH 'SEDITIOUS'

// Ends with  
WHERE n.name ENDS WITH 'ENGLAND'

// Regular expression
WHERE n.name =~ '.*SEDITIOUS.*LIBEL.*'
```

### Relationship Filtering

```cypher
// Filter by relationship type
MATCH (n)-[r:RELATED_TO]-(m)

// Filter by relationship properties
MATCH (n)-[r]-(m)
WHERE r.fact CONTAINS 'SEDITIOUS'

// Filter by multiple relationship types
MATCH (n)-[r:RELATED_TO|MENTIONED_IN|PART_OF]-(m)
```

### Combining Conditions

```cypher
MATCH (n)
WHERE (n.name CONTAINS 'SEDITIOUS' OR n.name CONTAINS 'LIBEL')
  AND n.name CONTAINS 'ENGLAND'
  AND NOT n.name CONTAINS 'CANADA'
OPTIONAL MATCH (n)-[e]-(m)
RETURN n, e, m
```

### Path Queries

```cypher
// Variable-length paths (1-3 hops)
MATCH path = (n)-[*1..3]-(m)
WHERE n.name CONTAINS 'SEDITIOUS'
RETURN path

// Shortest path
MATCH path = shortestPath((n)-[*]-(m))
WHERE n.name CONTAINS 'SEDITIOUS' 
  AND m.name CONTAINS 'ENGLAND'
RETURN path
```

## Practical Example Script

Save this as `cypher_query_falkordb.py`:

```python
#!/usr/bin/env python3
"""Execute Cypher queries on FalkorDB via Graphiti."""
import asyncio
import sys
from main import call_graphiti_tool, ensure_graphiti_session

async def execute_cypher(cypher_query: str):
    """Execute a Cypher query via Graphiti MCP."""
    await ensure_graphiti_session()
    
    print(f"Executing Cypher query:")
    print(f"{'='*80}")
    print(cypher_query)
    print(f"{'='*80}\n")
    
    try:
        # Try graph_query tool
        result, preview = await call_graphiti_tool(
            "graph_query",
            {
                "query": cypher_query
            }
        )
        
        print("✅ Query executed successfully!")
        print(f"\nResults:\n{result}")
        if preview:
            print(f"\nPreview:\n{preview[:500]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: The 'graph_query' tool may not be available.")
        print("Try using search_nodes or search_memory_facts instead.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default query for SEDITIOUS LIBEL
        query = """
        MATCH (n) 
        WHERE n.name CONTAINS 'SEDITIOUS' 
           OR n.name CONTAINS 'LIBEL'
           OR n.name CONTAINS 'RELATED OFFENCES'
        OPTIONAL MATCH (n)-[e]-(m) 
        RETURN n, e, m 
        LIMIT 100
        """
        print("No query provided, using default query for SEDITIOUS LIBEL...")
    else:
        # Read query from command line
        query = " ".join(sys.argv[1:])
    
    asyncio.run(execute_cypher(query))
```

## Usage Examples

```bash
# Use default query (SEDITIOUS LIBEL)
python cypher_query_falkordb.py

# Custom query
python cypher_query_falkordb.py "MATCH (n) WHERE n.name CONTAINS 'ENGLAND' RETURN n LIMIT 50"
```

## Note on Graphiti MCP Tools

Graphiti MCP provides these tools (check availability):
- `search_nodes` - Semantic search for nodes
- `search_memory_facts` - Search relationships/facts
- `get_episodes` - Get temporal episodes
- `get_nodes` - Get specific nodes
- `graph_query` - **May or may not be available** for direct Cypher queries

If `graph_query` is not available, use `search_nodes` with the text query instead:

```python
# Alternative: Use semantic search
nodes, preview = await call_graphiti_tool(
    "search_nodes",
    {
        "query": "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND",
        "group_ids": ["all_your_group_ids"],
        "max_nodes": 10
    }
)
```

## Quick Reference

| What You Want | Cypher Pattern |
|---------------|---------------|
| Find entity by name | `MATCH (n) WHERE n.name CONTAINS 'TERM' RETURN n` |
| Find with relationships | `MATCH (n)-[e]-(m) WHERE n.name CONTAINS 'TERM' RETURN n, e, m` |
| Case-insensitive | `WHERE toLower(n.name) CONTAINS 'term'` |
| Multiple terms (OR) | `WHERE n.name CONTAINS 'A' OR n.name CONTAINS 'B'` |
| Multiple terms (AND) | `WHERE n.name CONTAINS 'A' AND n.name CONTAINS 'B'` |
| Filter relationships | `MATCH (n)-[e]-(m) WHERE e.fact CONTAINS 'TERM'` |
| Limit results | `RETURN * LIMIT 100` |
| Get connected nodes | `MATCH (n)-[*1..2]-(m) WHERE n.name CONTAINS 'TERM' RETURN *` |



