# Entity Attribution Issue: Raymond Labarge

## Problem Summary
- **Entity**: "Raymond Labarge" appears in FalkorDB/Graphiti knowledge graph
- **Reported Location**: Associated with CHAPTER II
- **Actual Location**: Found in CHAPTER I, chunk 2 (line 97)
- **Search Results**: No search results found for this entity

## Findings

### 1. Document Verification ✅
- **Found in**: `documents/CHAPTER I.txt` at line 97
- **Context**: "Mr. Raymond Labarge, Deputy Minister of National Revenue, Customs and Excise Branch"
- **NOT found in**: CHAPTER II (verified with grep)

### 2. Chunking Verification ✅
- **Document**: CHAPTER I
- **Total Chunks**: 10 chunks
- **Entity Location**: Chunk 2 (word count: 445 words)
- **In Qdrant**: ✅ Chunk exists in Qdrant with correct doc_id and chunk_id

### 3. Search Issues ❌

#### Vector Search Problem
- Qdrant vector search for "Raymond Labarge" returned 5 results
- **NONE** of the top 5 results actually contain "Raymond Labarge"
- The chunk that DOES contain it (CHAPTER I, chunk 2) was not in the top results
- **Root Cause**: Embedding similarity mismatch - the entity name embedding doesn't match well with the chunk embedding

#### Graphiti Attribution Problem
- Entities lose `group_id` (document source) during normalization
- In `normalize_graphiti_results()`, entities only preserve: name, type, uuid
- **Missing**: group_id metadata that would indicate which document the entity came from
- This can lead to misattribution or confusion about document source

### 4. Code Issues Found

#### Missing group_id in Entity Normalization
```python
# In normalize_graphiti_results() - entities don't preserve group_id
entities.append({
    "name": node.get("name") or node.get("label"),
    "type": entity_type,
    "uuid": node.get("uuid")
    # ❌ Missing: group_id or document source
})
```

#### Search Query Issues
- Entity name alone ("Raymond Labarge") may not have good semantic match with chunk content
- The chunk contains a list of committee members, so the embedding is more about "committee members" than individual names
- Short entity queries vs. longer document chunks can have embedding mismatches

## Recommendations

### Immediate Fixes

1. **Fix Entity Attribution**
   - Extract and preserve `group_id` from Graphiti nodes
   - Add document source tracking to entities
   - Display correct document source in frontend

2. **Improve Vector Search**
   - Use hybrid search (keyword + semantic)
   - Add exact name matching boost
   - Query with more context (e.g., "Raymond Labarge Deputy Minister")

3. **Add Verification Layer**
   - Cross-check Graphiti entities against actual document content
   - Add validation that entities shown match document text
   - Display warnings for entities that can't be verified

### Long-term Improvements

1. **Entity Verification System**
   - Check if entities from Graphiti actually appear in source documents
   - Flag mismatches for review
   - Allow manual correction of misattributed entities

2. **Better Search Strategy**
   - Use keyword search as fallback when semantic search fails
   - Implement fuzzy name matching
   - Search across multiple name variations

3. **Transparency**
   - Show which document each entity came from
   - Display confidence scores for entity attribution
   - Allow users to see raw Graphiti results for debugging

## Current Status

- ✅ Entity exists in documents (CHAPTER I)
- ✅ Chunk is properly stored in Qdrant
- ❌ Vector search not finding entity
- ❌ Entity attribution may be incorrect
- ❌ No group_id tracking in entity normalization

## Next Steps

1. Fix `normalize_graphiti_results()` to preserve group_id
2. Add keyword fallback to vector search
3. Create entity verification endpoint
4. Test with "Raymond Labarge" query to verify fixes



