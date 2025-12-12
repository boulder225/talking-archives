#!/usr/bin/env python3
"""
Quick script to check Qdrant database status and contents.
"""
from qdrant_client import QdrantClient

def check_qdrant():
    qdrant = QdrantClient(host='localhost', port=6333)
    
    # Get collection info
    try:
        collection_info = qdrant.get_collection('archive_documents')
        print('=== QDRANT COLLECTION INFO ===')
        print(f'Collection name: archive_documents')
        print(f'Points count: {collection_info.points_count}')
        print(f'Status: {collection_info.status}')
        print(f'Vector size: {collection_info.config.params.vectors.size}')
        print(f'Distance: {collection_info.config.params.vectors.distance}')
    except Exception as e:
        print(f'Error getting collection: {e}')
        return
    
    # Get sample points
    print('\n=== SAMPLE POINTS ===')
    try:
        result = qdrant.scroll(
            collection_name='archive_documents',
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        points = result[0]
        print(f'Showing {len(points)} sample points:')
        for i, point in enumerate(points[:5], 1):
            payload = point.payload
            print(f'\n{i}. Point ID: {point.id}')
            print(f'   doc_id: {payload.get("doc_id")}')
            print(f'   chunk_id: {payload.get("chunk_id")}')
            print(f'   word_count: {payload.get("word_count")}')
            print(f'   char_count: {payload.get("char_count")}')
            text_preview = payload.get('text', '')[:100]
            print(f'   text preview: {text_preview}...')
    except Exception as e:
        print(f'Error getting points: {e}')
    
    # Count by doc_id
    print('\n=== DOCUMENTS SUMMARY ===')
    try:
        all_points = qdrant.scroll(
            collection_name='archive_documents',
            limit=10000,
            with_payload=True,
            with_vectors=False
        )[0]
        doc_counts = {}
        for point in all_points:
            doc_id = point.payload.get('doc_id')
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        for doc_id, count in sorted(doc_counts.items()):
            print(f'{doc_id}: {count} chunks')
        
        print(f'\nTotal: {len(all_points)} points across {len(doc_counts)} document(s)')
    except Exception as e:
        print(f'Error counting documents: {e}')

if __name__ == '__main__':
    check_qdrant()

