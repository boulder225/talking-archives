#!/usr/bin/env python3
"""
Query FalkorDB through Graphiti MCP tools.

Usage:
    python query_falkordb.py "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"
"""

import asyncio
import sys
from main import call_graphiti_tool, get_all_group_ids, ensure_graphiti_session

async def query_falkordb(query: str, group_ids: list = None, max_results: int = 10):
    """Query FalkorDB/Graphiti for nodes, facts, and episodes matching the query."""
    
    # Ensure Graphiti session is initialized
    await ensure_graphiti_session()
    
    # Get group_ids (all documents) if not specified
    if group_ids is None:
        group_ids = get_all_group_ids()
    
    print(f"Querying FalkorDB/Graphiti with: '{query}'")
    print(f"Searching in {len(group_ids)} document group(s): {', '.join(group_ids[:5])}{'...' if len(group_ids) > 5 else ''}")
    print("=" * 80)
    
    # 1. Search for nodes (entities)
    print("\n📊 SEARCHING NODES (Entities):")
    print("-" * 80)
    nodes_struct, nodes_preview = await call_graphiti_tool(
        "search_nodes",
        {
            "query": query,
            "group_ids": group_ids,
            "max_nodes": max_results
        }
    )
    
    # Parse and display nodes
    import json
    if isinstance(nodes_struct, str):
        try:
            nodes_data = json.loads(nodes_struct)
        except:
            nodes_data = {}
    else:
        nodes_data = nodes_struct or {}
    
    nodes = nodes_data.get("nodes", []) or []
    if nodes:
        print(f"Found {len(nodes)} node(s):")
        for i, node in enumerate(nodes[:max_results], 1):
            name = node.get("name") or node.get("label") or "Unknown"
            labels = node.get("labels", [])
            node_type = labels[0] if labels else node.get("attributes", {}).get("type", "Entity")
            print(f"  {i}. {name} ({node_type})")
            if node.get("attributes"):
                attrs = {k: v for k, v in node.get("attributes", {}).items() if k != "type"}
                if attrs:
                    print(f"     Attributes: {attrs}")
    else:
        print("  No nodes found matching the query.")
    
    # 2. Search for facts (relationships)
    print("\n🔗 SEARCHING FACTS (Relationships):")
    print("-" * 80)
    facts_struct, facts_preview = await call_graphiti_tool(
        "search_memory_facts",
        {
            "query": query,
            "group_ids": group_ids,
            "max_facts": max_results
        }
    )
    
    if isinstance(facts_struct, str):
        try:
            facts_data = json.loads(facts_struct)
        except:
            facts_data = {}
    else:
        facts_data = facts_struct or {}
    
    facts = facts_data.get("facts", []) or []
    if facts:
        print(f"Found {len(facts)} fact(s):")
        for i, fact in enumerate(facts[:max_results], 1):
            fact_text = fact.get("fact", "")
            fact_name = fact.get("name", "Unknown")
            valid_from = fact.get("valid_at")
            valid_to = fact.get("expired_at")
            print(f"  {i}. {fact_name}")
            print(f"     {fact_text[:200]}{'...' if len(fact_text) > 200 else ''}")
            if valid_from or valid_to:
                print(f"     Valid: {valid_from} to {valid_to}")
    else:
        print("  No facts found matching the query.")
    
    # 3. Get episodes (temporal events)
    print("\n📅 GETTING EPISODES (Temporal Events):")
    print("-" * 80)
    episodes_struct, episodes_preview = await call_graphiti_tool(
        "get_episodes",
        {
            "group_ids": group_ids,
            "max_episodes": max_results
        }
    )
    
    if isinstance(episodes_struct, str):
        try:
            episodes_data = json.loads(episodes_struct)
        except:
            episodes_data = {}
    else:
        episodes_data = episodes_struct or {}
    
    episodes = episodes_data.get("episodes", []) or []
    if episodes:
        print(f"Found {len(episodes)} episode(s):")
        for i, episode in enumerate(episodes[:max_results], 1):
            name = episode.get("name", "Unknown")
            content = episode.get("content") or episode.get("episode_body") or episode.get("text", "")
            ref_time = episode.get("reference_time") or episode.get("metadata", {}).get("reference_time")
            print(f"  {i}. {name}")
            if ref_time:
                print(f"     Time: {ref_time}")
            print(f"     {content[:200]}{'...' if len(content) > 200 else ''}")
    else:
        print("  No episodes found.")
    
    # Show preview text if available
    if nodes_preview or facts_preview or episodes_preview:
        print("\n📄 PREVIEW TEXT:")
        print("-" * 80)
        if nodes_preview:
            print(f"Nodes: {nodes_preview[:300]}...")
        if facts_preview:
            print(f"Facts: {facts_preview[:300]}...")
        if episodes_preview:
            print(f"Episodes: {episodes_preview[:300]}...")
    
    print("\n" + "=" * 80)
    print(f"✅ Query complete!")
    
    return {
        "nodes": nodes,
        "facts": facts,
        "episodes": episodes
    }

async def main():
    if len(sys.argv) < 2:
        query = "SEDITIOUS LIBEL AND RELATED OFFENCES IN ENGLAND"
        print(f"No query provided, using default: '{query}'")
        print("Usage: python query_falkordb.py 'YOUR QUERY HERE'")
        print()
    else:
        query = " ".join(sys.argv[1:])
    
    await query_falkordb(query)

if __name__ == "__main__":
    asyncio.run(main())



