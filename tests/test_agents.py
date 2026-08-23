from impact_engine import ImpactEngine
from signal_fusion import SignalFusionEngine
from models import CompanyProfile, CompanyProduct, Event, EventType, EventSource, Entity, EntityType, PriorityLevel

def test_impact_engine_scoring():
    profile = CompanyProfile(
        id="c1",
        company_name="Apex Therapeutics",
        profile_description="Targeted oncology company",
        therapeutic_areas=["Oncology"],
        geographic_markets=["EU", "USA"],
        products=[CompanyProduct(id="p1", drug_name="Apex-701", target="EGFR", indication="NSCLC", development_stage="Phase 2")],
        known_competitors=["OncoX Bio"]
    )

    evt = Event(
        id="e1",
        title="OncoX Bio EMA Marketing Application",
        event_type=EventType.REGULATORY_SUBMISSION,
        source=EventSource.SEC_FILINGS,
        summary="OncoX Bio filed EMA application in EU for EGFR inhibitor OncoX-201",
        jurisdiction="EU"
    )

    engine = ImpactEngine()
    score = engine.evaluate_event(evt, profile)

    assert score.relevance > 50.0
    assert score.priority in [PriorityLevel.HIGH, PriorityLevel.MEDIUM]
    assert "OncoX Bio" in score.relevance_reasoning

def test_signal_fusion_engine():
    evt1 = Event(id="e1", title="OncoX-201 Phase 3", event_type=EventType.PHASE_TRANSITION, source=EventSource.CLINICAL_TRIALS, summary="Phase 3 trial for OncoX-201.")
    evt2 = Event(id="e2", title="EuroPharma Licensing", event_type=EventType.PARTNERSHIP, source=EventSource.COMPANY_ANNOUNCEMENTS, summary="Licensed OncoX-201 in Europe.")

    fusion = SignalFusionEngine()
    signals = fusion.fuse_events_for_entity("OncoX-201", [evt1, evt2])

    assert len(signals) == 1
    assert "OncoX-201" in signals[0].title
    assert len(signals[0].observed_facts) == 2
