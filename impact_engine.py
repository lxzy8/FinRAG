from typing import List, Dict, Any, Optional
from models import Event, CompanyProfile, ImpactScore, PriorityLevel, EventType

class ImpactEngine:
    """Calculates Relevance, Impact, Urgency, and Confidence scores for an event relative to a Company Profile"""

    def evaluate_event(self, event: Event, profile: CompanyProfile) -> ImpactScore:
        """Evaluates relevance, impact, urgency, and confidence based on company profile context"""

        relevance = 10.0
        impact = 10.0
        urgency = 10.0
        confidence = 80.0

        rel_reasons = []
        imp_reasons = []
        urg_reasons = []
        conf_reasons = []

        # 1. Check entity overlap (Company, Drug, Target, Indication)
        event_text = f"{event.title} {event.summary} {event.details}".lower()

        # Check direct competitor match
        for competitor in profile.known_competitors:
            if competitor.lower() in event_text:
                relevance += 45.0
                rel_reasons.append(f"Involves direct competitor '{competitor}'")
                break

        # Check product / target / indication match
        for prod in profile.products:
            if prod.drug_name.lower() in event_text:
                relevance += 40.0
                rel_reasons.append(f"Directly mentions company product '{prod.drug_name}'")
            if prod.target and prod.target.lower() in event_text:
                relevance += 30.0
                rel_reasons.append(f"Matches drug target mechanism '{prod.target}'")
            if prod.indication.lower() in event_text:
                relevance += 20.0
                rel_reasons.append(f"Matches target indication '{prod.indication}'")

        # Check therapeutic area match
        for ta in profile.therapeutic_areas:
            if ta.lower() in event_text:
                relevance += 15.0
                rel_reasons.append(f"Aligns with core therapeutic area '{ta}'")

        # 2. Jurisdiction / Market evaluation
        jurisdiction_match = False
        if event.jurisdiction:
            event_jurisdictions = [j.strip().upper() for j in event.jurisdiction.replace(',', '/').split('/')]
            for market in profile.geographic_markets:
                if market.upper() in event_jurisdictions or market.upper() == "GLOBAL" or "GLOBAL" in event_jurisdictions:
                    jurisdiction_match = True
                    break
        else:
            jurisdiction_match = True

        if jurisdiction_match:
            relevance += 10.0
            urgency += 15.0
            urg_reasons.append(f"Applies to company active/target market jurisdiction ({event.jurisdiction})")
        else:
            relevance *= 0.5
            rel_reasons.append(f"Event jurisdiction ({event.jurisdiction}) outside primary markets")

        # 3. Event Type Impact & Urgency
        if event.event_type in [EventType.REGULATORY_APPROVAL, EventType.SAFETY_WARNING]:
            impact += 50.0
            urgency += 50.0
            imp_reasons.append(f"Critical regulatory event type ({event.event_type.value})")
            urg_reasons.append("Immediate compliance/regulatory attention required")
        elif event.event_type in [EventType.CLINICAL_RESULTS, EventType.PHASE_TRANSITION]:
            impact += 40.0
            urgency += 30.0
            imp_reasons.append(f"Significant clinical milestone ({event.event_type.value})")
        elif event.event_type in [EventType.REGULATORY_SUBMISSION, EventType.PARTNERSHIP, EventType.ACQUISITION]:
            impact += 35.0
            urgency += 25.0
            imp_reasons.append(f"Major strategic corporate activity ({event.event_type.value})")
        elif event.event_type in [EventType.PATENT_FILING, EventType.PUBLICATION]:
            impact += 20.0
            urgency += 15.0
            imp_reasons.append(f"IP/Scientific disclosure ({event.event_type.value})")

        # 4. Source Confidence
        if event.source.value in ["Regulatory Authority", "ClinicalTrials.gov", "Company Filings (SEC)"]:
            confidence += 15.0
            conf_reasons.append(f"Official regulatory/primary regulatory source ({event.source.value})")
        else:
            confidence += 5.0
            conf_reasons.append(f"Published secondary media/news source ({event.source.value})")

        # Clamp scores to 0-100
        relevance = min(100.0, max(0.0, relevance))
        impact = min(100.0, max(0.0, impact))
        urgency = min(100.0, max(0.0, urgency))
        confidence = min(100.0, max(0.0, confidence))

        # Assign priority based on composite score
        composite = (relevance * 0.4) + (impact * 0.4) + (urgency * 0.2)
        if composite >= 70.0:
            priority = PriorityLevel.HIGH
        elif composite >= 40.0:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW

        return ImpactScore(
            relevance=round(relevance, 1),
            impact=round(impact, 1),
            urgency=round(urgency, 1),
            confidence=round(confidence, 1),
            priority=priority,
            relevance_reasoning="; ".join(rel_reasons) or "General landscape item",
            impact_reasoning="; ".join(imp_reasons) or "Standard developments",
            urgency_reasoning="; ".join(urg_reasons) or "Routine monitoring",
            confidence_reasoning="; ".join(conf_reasons) or "Standard confidence"
        )
