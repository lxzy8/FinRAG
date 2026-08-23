from typing import Dict, List, Any, Set, Optional
from datetime import datetime
from models import Entity, Event, EntityType

class KnowledgeGraph:
    """In-memory graph storing entities, events, historical relationships, and state transitions"""

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []

    def add_entity_node(self, entity: Entity):
        """Adds or updates an entity node in the graph"""
        if entity.id not in self.nodes:
            self.nodes[entity.id] = {
                "id": entity.id,
                "label": entity.name,
                "type": "ENTITY",
                "entity_type": entity.entity_type.value,
                "synonyms": entity.synonyms,
                "metadata": entity.metadata
            }

    def add_event_node(self, event: Event):
        """Adds an event node to the graph and connects involved entities"""
        if event.id not in self.nodes:
            self.nodes[event.id] = {
                "id": event.id,
                "label": event.title,
                "type": "EVENT",
                "event_type": event.event_type.value,
                "source": event.source.value,
                "timestamp": event.timestamp.isoformat(),
                "summary": event.summary,
                "jurisdiction": event.jurisdiction
            }

            # Record state history
            self.history.append({
                "action": "ADD_EVENT",
                "event_id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "title": event.title
            })

            # Connect event to involved entities
            for entity in event.entities:
                self.add_entity_node(entity)
                self.add_edge(
                    source_id=event.id,
                    target_id=entity.id,
                    relation_type="INVOLVES_ENTITY",
                    properties={"timestamp": event.timestamp.isoformat()}
                )

    def add_edge(self, source_id: str, target_id: str, relation_type: str, properties: Optional[Dict[str, Any]] = None):
        """Adds a directional relationship edge between two nodes"""
        edge = {
            "source": source_id,
            "target": target_id,
            "relation": relation_type,
            "properties": properties or {}
        }
        if edge not in self.edges:
            self.edges.append(edge)

    def add_domain_relationships(self, source_entity: Entity, relation_type: str, target_entity: Entity):
        """Adds domain-specific biopharma relationships (e.g. Drug -> targets -> Target)"""
        self.add_entity_node(source_entity)
        self.add_entity_node(target_entity)
        self.add_edge(source_entity.id, target_entity.id, relation_type)

    def get_neighbors(self, node_id: str, relation_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves neighboring nodes for a given node"""
        neighbors = []
        for edge in self.edges:
            if edge["source"] == node_id:
                if not relation_filter or edge["relation"] == relation_filter:
                    target_node = self.nodes.get(edge["target"])
                    if target_node:
                        neighbors.append({"node": target_node, "relation": edge["relation"], "direction": "OUT"})
            elif edge["target"] == node_id:
                if not relation_filter or edge["relation"] == relation_filter:
                    source_node = self.nodes.get(edge["source"])
                    if source_node:
                        neighbors.append({"node": source_node, "relation": edge["relation"], "direction": "IN"})
        return neighbors

    def multi_hop_search(self, start_entity_name: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Performs a multi-hop traversal starting from an entity name"""
        matching_nodes = [
            nid for nid, n in self.nodes.items()
            if start_entity_name.lower() in n.get("label", "").lower()
        ]

        visited: Set[str] = set()
        results: List[Dict[str, Any]] = []

        def dfs(current_id: str, depth: int, path: List[str]):
            if depth > max_hops or current_id in visited:
                return
            visited.add(current_id)
            node = self.nodes.get(current_id)
            if node:
                results.append({"node": node, "path": path, "depth": depth})

            for edge in self.edges:
                next_id = None
                rel = edge["relation"]
                if edge["source"] == current_id:
                    next_id = edge["target"]
                elif edge["target"] == current_id:
                    next_id = edge["source"]

                if next_id and next_id not in visited:
                    dfs(next_id, depth + 1, path + [f"--[{rel}]-->", next_id])

        for start_id in matching_nodes:
            dfs(start_id, 0, [start_id])

        return results

    def get_historical_timeline(self, entity_id: str) -> List[Dict[str, Any]]:
        """Extracts chronological events connected to an entity"""
        timeline = []
        for edge in self.edges:
            if edge["target"] == entity_id or edge["source"] == entity_id:
                other_id = edge["target"] if edge["source"] == entity_id else edge["source"]
                other_node = self.nodes.get(other_id)
                if other_node and other_node.get("type") == "EVENT":
                    timeline.append(other_node)

        # Sort by timestamp
        timeline.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return timeline
