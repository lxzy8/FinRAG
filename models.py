from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    COMPANY = "Company"
    DRUG = "Drug"
    TARGET = "Target"
    DISEASE = "Disease"
    INDICATION = "Indication"
    CLINICAL_TRIAL = "Clinical Trial"
    PATENT = "Patent"
    REGULATOR = "Regulator"
    JURISDICTION = "Jurisdiction"
    RESEARCHER = "Researcher"
    ORGANIZATION = "Organization"

class Entity(BaseModel):
    id: str
    name: str
    entity_type: EntityType
    synonyms: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EventType(str, Enum):
    TRIAL_INITIATION = "Clinical-trial initiation"
    PHASE_TRANSITION = "Phase transition"
    TRIAL_STATUS_CHANGE = "Trial-status change"
    ENDPOINT_PROTOCOL_CHANGE = "Endpoint/protocol change"
    CLINICAL_RESULTS = "Clinical results"
    REGULATORY_SUBMISSION = "Regulatory submission"
    REGULATORY_APPROVAL = "Regulatory approval"
    SAFETY_WARNING = "Safety warning"
    GUIDANCE_CHANGE = "Guidance change"
    PATENT_FILING = "Patent filing"
    PATENT_EVENT = "Patent event"
    PUBLICATION = "Publication"
    PARTNERSHIP = "Partnership"
    LICENSING_DEAL = "Licensing deal"
    ACQUISITION = "Acquisition"
    PIPELINE_CHANGE = "Company pipeline change"
    MARKET_LAUNCH_ACTIVITY = "Market/launch activity"

class EventSource(str, Enum):
    PUBMED = "PubMed"
    CLINICAL_TRIALS = "ClinicalTrials.gov"
    SEC_FILINGS = "Company Filings (SEC)"
    COMPANY_ANNOUNCEMENTS = "Company Announcements"
    REGULATORY_BODY = "Regulatory Authority"
    PATENT_OFFICE = "Patent Office"
    NEWS = "Biotech/Scientific News"

class PriorityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ImpactScore(BaseModel):
    relevance: float = Field(ge=0.0, le=100.0, description="Relevance score (0-100)")
    impact: float = Field(ge=0.0, le=100.0, description="Impact score (0-100)")
    urgency: float = Field(ge=0.0, le=100.0, description="Urgency score (0-100)")
    confidence: float = Field(ge=0.0, le=100.0, description="Confidence score (0-100)")
    priority: PriorityLevel = PriorityLevel.MEDIUM
    relevance_reasoning: str = ""
    impact_reasoning: str = ""
    urgency_reasoning: str = ""
    confidence_reasoning: str = ""

class Event(BaseModel):
    id: str
    title: str
    event_type: EventType
    source: EventSource
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entities: List[Entity] = Field(default_factory=list)
    summary: str
    details: str = ""
    source_url: Optional[str] = None
    raw_content: Optional[str] = None
    jurisdiction: Optional[str] = None
    historical_version: int = 1

class CompanyProduct(BaseModel):
    id: str
    drug_name: str
    target: Optional[str] = None
    indication: str
    development_stage: str
    geographic_markets: List[str] = Field(default_factory=list)

class CompanyProfile(BaseModel):
    id: str
    company_name: str
    profile_description: str
    therapeutic_areas: List[str] = Field(default_factory=list)
    geographic_markets: List[str] = Field(default_factory=list)
    products: List[CompanyProduct] = Field(default_factory=list)
    known_competitors: List[str] = Field(default_factory=list)
    strategic_objectives: List[str] = Field(default_factory=list)

class EvidenceItem(BaseModel):
    id: str
    source: EventSource
    title: str
    excerpt: str
    url: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 1.0

class FusedSignal(BaseModel):
    id: str
    title: str
    summary: str
    observed_facts: List[str] = Field(default_factory=list)
    evidence_supported_interpretations: List[str] = Field(default_factory=list)
    model_generated_inferences: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=100.0, default=85.0)
    related_events: List[Event] = Field(default_factory=list)
    related_entities: List[Entity] = Field(default_factory=list)
    priority: PriorityLevel = PriorityLevel.MEDIUM
    impact_score: Optional[ImpactScore] = None

class IntelligenceAlert(BaseModel):
    id: str
    title: str
    summary: str
    priority: PriorityLevel
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    affected_product: Optional[str] = None
    affected_market: Optional[str] = None
    event: Optional[Event] = None
    fused_signal: Optional[FusedSignal] = None
    impact_score: Optional[ImpactScore] = None
    recommended_action: str = ""
