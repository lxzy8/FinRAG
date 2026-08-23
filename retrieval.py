from typing import List, Dict, Any, Optional
from models import Event, EvidenceItem, EventSource
from knowledge_graph import KnowledgeGraph

class HybridRetriever:
    """Combines vector semantic search over event contents with structured Knowledge Graph multi-hop retrieval"""

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.events_db: Dict[str, Event] = {}

    def index_events(self, events: List[Event]):
        """Index events into in-memory store and Knowledge Graph"""
        for event in events:
            self.events_db[event.id] = event
            self.kg.add_event_node(event)

    def vector_search(self, query: str, top_k: int = 5) -> List[Event]:
        """Performs semantic/keyword score matching over indexed events"""
        query_terms = [t.lower() for t in query.split() if len(t) >= 2]
        scored_events = []

        for event in self.events_db.values():
            text_corpus = f"{event.title} {event.summary} {event.details} {' '.join([e.name for e in event.entities])}".lower()
            score = 0
            for term in query_terms:
                if term in text_corpus:
                    score += 1
            if score > 0:
                scored_events.append((score, event))

        scored_events.sort(key=lambda x: x[0], reverse=True)
        return [e[1] for e in scored_events[:top_k]]

    def graph_search(self, query: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Retrieves multi-hop entity/event structures from the Knowledge Graph"""
        return self.kg.multi_hop_search(query, max_hops=max_hops)

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Combines semantic vector results and multi-hop graph paths into evidence payload"""
        vector_results = self.vector_search(query, top_k=top_k)
        graph_results = self.graph_search(query)

        evidence_items: List[EvidenceItem] = []
        for evt in vector_results:
            evidence_items.append(
                EvidenceItem(
                    id=f"ev_{evt.id}",
                    source=evt.source,
                    title=evt.title,
                    excerpt=evt.summary,
                    url=evt.source_url,
                    timestamp=evt.timestamp,
                    confidence=0.9
                )
            )

        return {
            "query": query,
            "vector_events": vector_results,
            "graph_paths": graph_results,
            "evidence": evidence_items,
            "total_matches": len(vector_results) + len(graph_results)
        }
