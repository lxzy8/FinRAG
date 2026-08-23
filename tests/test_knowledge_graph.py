from knowledge_graph import KnowledgeGraph
from models import Entity, EntityType, Event, EventType, EventSource
from datetime import datetime

def test_knowledge_graph_traversals():
    kg = KnowledgeGraph()

    company = Entity(id="comp_1", name="OncoX Bio", entity_type=EntityType.COMPANY)
    drug = Entity(id="drug_1", name="OncoX-201", entity_type=EntityType.DRUG)
    target = Entity(id="target_1", name="EGFR", entity_type=EntityType.TARGET)

    kg.add_domain_relationships(company, "DEVELOPED", drug)
    kg.add_domain_relationships(drug, "TARGETS", target)

    event = Event(
        id="evt_100",
        title="Phase 3 Results",
        event_type=EventType.CLINICAL_RESULTS,
        source=EventSource.CLINICAL_TRIALS,
        entities=[company, drug],
        summary="Positive results."
    )
    kg.add_event_node(event)

    neighbors = kg.get_neighbors("comp_1")
    assert len(neighbors) > 0

    multihop = kg.multi_hop_search("OncoX", max_hops=2)
    assert len(multihop) > 0
