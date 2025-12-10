#!/usr/bin/env python3
"""Execute Cypher queries on FalkorDB via Graphiti MCP or direct connection."""
import asyncio
import sys
from main import call_graphiti_tool, ensure_graphiti_session, get_all_group_ids

async def execute_cypher_via_graphiti(cypher_query: str):
    """Try to execute Cypher query via Graphiti MCP graph_query tool."""
    await ensure_graphiti_session()
    
    print(f"Attempting to execute Cypher query via Graphiti MCP...")
    print(f"{'='*80}")
    print(cypher_query.strip())
    print(f"{'='*80}\n")
    
    try:
        # Try graph_query tool (may not be available)
        result, preview = await call_graphiti_tool(
            "graph_query",
            {
                "query": cypher_query
            }
        )
        
        print("✅ Query executed successfully via Graphiti!")
        print(f"\nResults:\n{result}")
        if preview:
            print(f"\nPreview:\n{preview[:500]}...")
        return result
            
    except Exception as e:
        print(f"❌ Graphiti graph_query tool not available or error: {e}")
        print("\n📝 Note: Direct Cypher queries may not be supported via Graphiti MCP.")
        print("   Try using search_nodes or search_memory_facts instead.")
        return None

async def search_via_semantic_tools(search_term: str):
    """Fallback: Use Graphiti's semantic search tools instead of Cypher."""
    await ensure_graphiti_session()
    group_ids = get_all_group_ids()
    
    print(f"\nUsing semantic search as alternative to Cypher query...")
    print(f"Search term: '{search_term}'\n")
    
    # Search nodes
    print("📊 Searching nodes:")
    nodes_struct, nodes_preview = await call_graphiti_tool(
        "search_nodes",
        {
            "query": search_term,
            "group_ids": group_ids,
            "max_nodes": 20
        }
    )
    
    # Search facts
    print("🔗 Searching facts/relationships:")
    facts_struct, facts_preview = await call_graphiti_tool(
        "search_memory_facts",
        {
            "query": search_term,
            "group_ids": group_ids,
            "max_facts": 20
        }
    )
    
    print(f"\nNodes preview: {nodes_preview[:300] if nodes_preview else 'None'}...")
    print(f"Facts preview: {facts_preview[:300] if facts_preview else 'None'}...")
    
    return nodes_struct, facts_struct

def generate_cypher_from_search_term(search_term: str):
    """Generate a Cypher query from a search term."""
    # Extract key terms
    terms = search_term.upper().split()
    # Remove common words
    stopwords = {'AND', 'OR', 'THE', 'A', 'AN', 'IN', 'ON', 'AT', 'TO', 'FOR', 'OF', 'WITH'}
    search_terms = [t for t in terms if t not in stopwords and len(t) > 2]
    
    # Build WHERE clause
    conditions = []
    for term in search_terms[:5]:  # Limit to 5 terms to avoid too long query
        conditions.append(f"n.name CONTAINS '{term}'")
    
    where_clause = " OR ".join(conditions)
    
    cypher = f"""
MATCH (n)
WHERE {where_clause}
OPTIONAL MATCH (n)-[e]-(m)
RETURN n, e, m
LIMIT 100
"""
    return cypher.strip()

async def main():
    if len(sys.argv) < 2:
        search_term = "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"
        print("No query provided, using default search term...")
        print(f"Search term: '{search_term}'\n")
        
        # Try Cypher first
        cypher_query = generate_cypher_from_search_term(search_term)
        result = await execute_cypher_via_graphiti(cypher_query)
        
        # If Cypher doesn't work, use semantic search
        if result is None:
            await search_via_semantic_tools(search_term)
    else:
        query_input = " ".join(sys.argv[1:])
        
        # Check if it looks like Cypher query
        if query_input.strip().upper().startswith("MATCH"):
            # Direct Cypher query
            result = await execute_cypher_via_graphiti(query_input)
            if result is None:
                print("\n💡 Tip: Try using search_nodes or search_memory_facts tools instead.")
        else:
            # Text search - try Cypher first, then semantic
            cypher_query = generate_cypher_from_search_term(query_input)
            result = await execute_cypher_via_graphiti(cypher_query)
            
            if result is None:
                await search_via_semantic_tools(query_input)

if __name__ == "__main__":
    asyncio.run(main())


