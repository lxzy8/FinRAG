import uuid
from typing import List, Dict, Any
from models import Event, CompanyProfile, IntelligenceAlert, PriorityLevel
from impact_engine import ImpactEngine
from signal_fusion import SignalFusionEngine

class DailyIntelligenceAgent:
    """Continuously monitors incoming biopharma events, asks 'What changed?', assesses relevance, and generates alerts"""

    def __init__(self):
        self.impact_engine = ImpactEngine()
        self.fusion_engine = SignalFusionEngine()

    def process_incoming_events(self, events: List[Event], profile: CompanyProfile) -> List[IntelligenceAlert]:
        """Evaluates batch of incoming events against company profile and returns prioritized alerts"""
        alerts = []

        for event in events:
            score = self.impact_engine.evaluate_event(event, profile)

            # Generate alert for items with non-negligible priority
            if score.priority in [PriorityLevel.HIGH, PriorityLevel.MEDIUM]:
                alert_id = f"alert_{uuid.uuid4().hex[:8]}"

                rec_action = "Monitor trial progress and update competitor matrix."
                if "EMA" in event.title or "EU" in event.jurisdiction:
                    rec_action = "Review European regulatory filing timelines and commercial supply chain."
                elif "Phase 3" in event.title or "Phase 3" in event.summary:
                    rec_action = "Assess potential market overlap and prepare competitive differentiation analysis."

                alert = IntelligenceAlert(
                    id=alert_id,
                    title=f"[{score.priority.value} PRIORITY] {event.title}",
                    summary=f"WHAT CHANGED: {event.summary}\n\nWHY IT MATTERS: {score.relevance_reasoning}. {score.impact_reasoning}.",
                    priority=score.priority,
                    timestamp=event.timestamp,
                    affected_product=profile.products[0].drug_name if profile.products else None,
                    affected_market=event.jurisdiction or profile.geographic_markets[0] if profile.geographic_markets else None,
                    event=event,
                    impact_score=score,
                    recommended_action=rec_action
                )
                alerts.append(alert)

        # Sort alerts by priority (HIGH first, then MEDIUM)
        priority_order = {PriorityLevel.HIGH: 0, PriorityLevel.MEDIUM: 1, PriorityLevel.LOW: 2}
        alerts.sort(key=lambda a: priority_order[a.priority])

        return alerts
