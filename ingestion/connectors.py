import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from models import Event, EventType, EventSource, Entity, EntityType

class BaseConnector:
    """Base connector interface for biopharma data sources"""
    def __init__(self, source_name: EventSource):
        self.source_name = source_name

    def fetch_recent_data(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError

class ClinicalTrialsConnector(BaseConnector):
    """Connector for ClinicalTrials.gov API / datasets"""
    def __init__(self):
        super().__init__(EventSource.CLINICAL_TRIALS)

    def fetch_recent_data(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Mock/Sample data representing real ClinicalTrials API responses
        mock_records = [
            {
                "id": "NCT05891234",
                "title": "Phase 3 Study of OncoX-201 vs Pembrolizumab in Non-Small Cell Lung Cancer",
                "event_type": EventType.PHASE_TRANSITION,
                "summary": "OncoX Bio initiated a global Phase 3 study (NCT05891234) evaluating OncoX-201 in combination with chemotherapy for advanced NSCLC across 120 global sites.",
                "details": "Sponsor: OncoX Bio. Drug: OncoX-201 (EGFR/HER2 inhibitor). Target: EGFR. Indication: Non-Small Cell Lung Cancer. Primary Endpoint: Progression Free Survival (PFS). Phase: Phase 3. Estimated Completion: Dec 2026.",
                "jurisdiction": "USA, EU, Japan",
                "entities": [
                    {"name": "OncoX Bio", "type": EntityType.COMPANY},
                    {"name": "OncoX-201", "type": EntityType.DRUG},
                    {"name": "EGFR", "type": EntityType.TARGET},
                    {"name": "Non-Small Cell Lung Cancer", "type": EntityType.DISEASE},
                    {"name": "NCT05891234", "type": EntityType.CLINICAL_TRIAL}
                ],
                "source_url": "https://clinicaltrials.gov/ct2/show/NCT05891234",
                "timestamp": datetime.utcnow()
            },
            {
                "id": "NCT04910293",
                "title": "Phase 2 Study of Neuro-88 for Alzheimer's Disease Reaches Primary Endpoint",
                "event_type": EventType.CLINICAL_RESULTS,
                "summary": "NeuroGen Therapeutics reported positive Phase 2 results for Neuro-88 in early Alzheimer's disease showing significant reduction in amyloid plaques.",
                "details": "Sponsor: NeuroGen Therapeutics. Drug: Neuro-88. Indication: Alzheimer's Disease. Primary Endpoint met with p < 0.001 statistically significant reduction.",
                "jurisdiction": "USA",
                "entities": [
                    {"name": "NeuroGen Therapeutics", "type": EntityType.COMPANY},
                    {"name": "Neuro-88", "type": EntityType.DRUG},
                    {"name": "Amyloid Beta", "type": EntityType.TARGET},
                    {"name": "Alzheimer's Disease", "type": EntityType.DISEASE},
                    {"name": "NCT04910293", "type": EntityType.CLINICAL_TRIAL}
                ],
                "source_url": "https://clinicaltrials.gov/ct2/show/NCT04910293",
                "timestamp": datetime.utcnow()
            }
        ]
        return mock_records[:limit]

class PubMedConnector(BaseConnector):
    """Connector for PubMed / Scientific publications"""
    def __init__(self):
        super().__init__(EventSource.PUBMED)

    def fetch_recent_data(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        mock_records = [
            {
                "id": "PMID38291023",
                "title": "Structural basis of dual EGFR/HER2 inhibition by next-generation small molecule OncoX-201",
                "event_type": EventType.PUBLICATION,
                "summary": "New publication in Nature Medicine detailing cryo-EM structures of OncoX-201 bound to mutant EGFR variants resistant to osimertinib.",
                "details": "Authors: Smith et al. Journal: Nature Medicine. Key Findings: OncoX-201 overcomes exon 20 insertion and C797S resistance mutations in vitro and in xenograft models.",
                "jurisdiction": "Global",
                "entities": [
                    {"name": "OncoX-201", "type": EntityType.DRUG},
                    {"name": "EGFR", "type": EntityType.TARGET},
                    {"name": "HER2", "type": EntityType.TARGET},
                    {"name": "Non-Small Cell Lung Cancer", "type": EntityType.DISEASE}
                ],
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/38291023/",
                "timestamp": datetime.utcnow()
            }
        ]
        return mock_records[:limit]

class CompanyFilingsConnector(BaseConnector):
    """Connector for SEC 10-K / 8-K / Investor Filings"""
    def __init__(self):
        super().__init__(EventSource.SEC_FILINGS)

    def fetch_recent_data(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        mock_records = [
            {
                "id": "SEC-8K-ONCOX-2024",
                "title": "OncoX Bio SEC Form 8-K: Expansion of Commercial Manufacturing and EMA Regulatory Strategy",
                "event_type": EventType.REGULATORY_SUBMISSION,
                "summary": "OncoX Bio filed Form 8-K disclosing lease agreement for commercial manufacturing facility in Ireland and planned EMA Marketing Authorization Application (MAA) submission in Q4.",
                "details": "SEC Form 8-K Filing. Highlights: Facility lease in Dublin, Ireland for EU commercial supply. MAA submission targeted for Q4 2024 following Phase 3 readout.",
                "jurisdiction": "EU, USA",
                "entities": [
                    {"name": "OncoX Bio", "type": EntityType.COMPANY},
                    {"name": "EMA", "type": EntityType.REGULATOR},
                    {"name": "Europe", "type": EntityType.JURISDICTION}
                ],
                "source_url": "https://www.sec.gov/edgar/searchedgar/companysearch",
                "timestamp": datetime.utcnow()
            }
        ]
        return mock_records[:limit]

class AnnouncementsConnector(BaseConnector):
    """Connector for Press Releases & Corporate Announcements"""
    def __init__(self):
        super().__init__(EventSource.COMPANY_ANNOUNCEMENTS)

    def fetch_recent_data(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        mock_records = [
            {
                "id": "PR-ONCOX-EU-PARTNER",
                "title": "OncoX Bio Announces Strategic Licensing Partnership with EuroPharma for European Distribution",
                "event_type": EventType.PARTNERSHIP,
                "summary": "OncoX Bio entered an exclusive $450M commercial licensing agreement with EuroPharma to commercialize OncoX-201 in the European Union.",
                "details": "Deal Terms: $50M upfront, $400M milestone payments plus tiered royalties. EuroPharma handles EU regulatory filings and commercialization.",
                "jurisdiction": "EU",
                "entities": [
                    {"name": "OncoX Bio", "type": EntityType.COMPANY},
                    {"name": "EuroPharma", "type": EntityType.COMPANY},
                    {"name": "OncoX-201", "type": EntityType.DRUG},
                    {"name": "EU", "type": EntityType.JURISDICTION}
                ],
                "source_url": "https://press.oncoxbio.com/2024-eu-partnership",
                "timestamp": datetime.utcnow()
            }
        ]
        return mock_records[:limit]

class IngestionManager:
    """Manager to coordinate all source connectors"""
    def __init__(self):
        self.connectors = [
            ClinicalTrialsConnector(),
            PubMedConnector(),
            CompanyFilingsConnector(),
            AnnouncementsConnector()
        ]

    def fetch_all(self, query: str = "*", limit_per_source: int = 5) -> List[Dict[str, Any]]:
        all_data = []
        for conn in self.connectors:
            try:
                data = conn.fetch_recent_data(query, limit=limit_per_source)
                for item in data:
                    item["source_name"] = conn.source_name
                all_data.extend(data)
            except Exception as e:
                print(f"Error fetching from {conn.source_name}: {e}")
        return all_data
