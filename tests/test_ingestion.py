from ingestion.connectors import IngestionManager
from ingestion.extraction import ExtractionPipeline
from models import EventType, EventSource

def test_ingestion_manager_fetch():
    mgr = IngestionManager()
    data = mgr.fetch_all(limit_per_source=2)
    assert len(data) > 0
    assert any("title" in item for item in data)

def test_extraction_pipeline():
    pipeline = ExtractionPipeline()
    raw_item = {
        "id": "test_nct_001",
        "title": "Phase 3 Trial of TestDrug in NSCLC",
        "event_type": EventType.PHASE_TRANSITION,
        "source_name": EventSource.CLINICAL_TRIALS,
        "summary": "Phase 3 trial initiated by TestCorp.",
        "entities": [
            {"name": "TestCorp", "type": "Company"},
            {"name": "TestDrug", "type": "Drug"}
        ]
    }
    event, is_new = pipeline.process_raw_item(raw_item)
    assert is_new is True
    assert event.id == "test_nct_001"
    assert len(event.entities) == 2
    assert event.entities[0].name == "TestCorp"
