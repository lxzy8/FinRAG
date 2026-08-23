from models import CompanyProfile, CompanyProduct, Entity, EntityType, Event, EventType, EventSource, ImpactScore, PriorityLevel

def test_company_profile_instantiation():
    profile = CompanyProfile(
        id="c1",
        company_name="BioTest Corp",
        profile_description="Test biotech company",
        therapeutic_areas=["Oncology"],
        geographic_markets=["USA"],
        products=[
            CompanyProduct(
                id="p1",
                drug_name="TestDrug-101",
                indication="NSCLC",
                development_stage="Phase 2"
            )
        ],
        known_competitors=["CompBio"],
        strategic_objectives=["Phase 3 launch"]
    )
    assert profile.company_name == "BioTest Corp"
    assert len(profile.products) == 1
    assert profile.products[0].drug_name == "TestDrug-101"

def test_entity_and_event_models():
    entity = Entity(id="ent_1", name="OncoX Bio", entity_type=EntityType.COMPANY)
    event = Event(
        id="evt_1",
        title="Phase 3 Study Initiated",
        event_type=EventType.TRIAL_INITIATION,
        source=EventSource.CLINICAL_TRIALS,
        entities=[entity],
        summary="Trial initiated."
    )
    assert event.id == "evt_1"
    assert event.entities[0].name == "OncoX Bio"
