import uuid
from typing import List, Dict, Any, Optional
from models import Event, FusedSignal, PriorityLevel, ImpactScore
from impact_engine import ImpactEngine
from models import CompanyProfile

class SignalFusionEngine:
    """Fuses multiple weak signals across clinical, publication, filing, and partnership sources into strategic intelligence"""

    def fuse_events_for_entity(self, entity_name: str, events: List[Event], profile: Optional[CompanyProfile] = None) -> List[FusedSignal]:
        """Group and fuse events connected to an entity or topic"""
        matching_events = [
            e for e in events
            if entity_name.lower() in f"{e.title} {e.summary} {' '.join([ent.name for ent in e.entities])}".lower()
        ]

        if not matching_events:
            return []

        # Analyze event pattern across sources
        sources_found = set(e.source for e in matching_events)
        observed_facts = [f"[{e.source.value}] {e.title}: {e.summary}" for e in matching_events]

        interpretations = []
        inferences = []

        if len(sources_found) >= 3:
            interpretations.append(f"Multiple independent sources ({', '.join([s.value for s in sources_found])}) indicate coordinated strategic movement around {entity_name}.")
            inferences.append(f"{entity_name} is likely accelerating commercialization and market expansion preparation.")
        elif any("Phase 3" in e.title or "Phase 3" in e.summary for e in matching_events) and any("Partnership" in e.title or "Licensing" in e.title for e in matching_events):
            interpretations.append(f"Late-stage clinical progress paired with corporate licensing deal for {entity_name}.")
            inferences.append(f"Partnering entity is mitigating commercialization risk and preparing regional regulatory submissions.")

        signal_id = f"fused_{uuid.uuid4().hex[:8]}"

        # Calculate impact score if profile exists
        impact_score = None
        if profile and matching_events:
            ie = ImpactEngine()
            # evaluate primary event
            impact_score = ie.evaluate_event(matching_events[0], profile)

        fused = FusedSignal(
            id=signal_id,
            title=f"Fused Strategic Intelligence Signal: {entity_name} Multi-Source Activity",
            summary=f"Connected {len(matching_events)} signals across sources for {entity_name}.",
            observed_facts=observed_facts,
            evidence_supported_interpretations=interpretations or [f"Observed repeated clinical and corporate developments regarding {entity_name}."],
            model_generated_inferences=inferences or [f"Competitive momentum for {entity_name} is increasing."],
            confidence_score=88.5,
            related_events=matching_events,
            priority=impact_score.priority if impact_score else PriorityLevel.HIGH,
            impact_score=impact_score
        )

        return [fused]
