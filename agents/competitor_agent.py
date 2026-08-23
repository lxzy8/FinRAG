from typing import List, Dict, Any, Optional
from models import Event, CompanyProfile, FusedSignal
from knowledge_graph import KnowledgeGraph
from retrieval import HybridRetriever
from signal_fusion import SignalFusionEngine
from impact_engine import ImpactEngine

class CompetitorIntelligenceAgent:
    """Agent dedicated to searching, connecting, and assessing competitor activities across sources"""

    def __init__(self, knowledge_graph: KnowledgeGraph, retriever: HybridRetriever):
        self.kg = knowledge_graph
        self.retriever = retriever
        self.fusion_engine = SignalFusionEngine()
        self.impact_engine = ImpactEngine()

    def analyze_competitor(self, competitor_name: str, customer_profile: CompanyProfile) -> Dict[str, Any]:
        """Conducts a multi-source competitor intelligence assessment"""
        # Retrieve vector events and graph structure
        retrieval_res = self.retriever.hybrid_retrieve(competitor_name, top_k=10)
        events: List[Event] = retrieval_res["vector_events"]

        # Retrieve historical timeline from Knowledge Graph
        timeline = self.kg.get_historical_timeline(competitor_name.lower().replace(" ", "_"))
        if not timeline and events:
            timeline = [
                {
                    "id": e.id,
                    "label": e.title,
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp.isoformat(),
                    "summary": e.summary
                }
                for e in events
            ]

        # Signal fusion
        fused_signals = self.fusion_engine.fuse_events_for_entity(
            competitor_name, events, profile=customer_profile
        )

        # Impact evaluation
        evaluated_events = []
        for evt in events:
            score = self.impact_engine.evaluate_event(evt, customer_profile)
            evaluated_events.append({"event": evt, "impact_score": score})

        # Summary reasoning
        strategic_summary = (
            f"Competitor Intelligence Analysis for '{competitor_name}':\n"
            f"- Discovered {len(events)} events across clinical trials, SEC filings, publications, and announcements.\n"
            f"- Identified {len(fused_signals)} cross-source fused intelligence signals.\n"
            f"- Competitive Threat Level: {'HIGH' if any(s.get('impact_score') and s['impact_score'].priority.value == 'HIGH' for s in evaluated_events) else 'MEDIUM'}."
        )

        return {
            "competitor_name": competitor_name,
            "strategic_summary": strategic_summary,
            "events_analyzed": len(events),
            "timeline": timeline,
            "fused_signals": fused_signals,
            "evaluated_events": evaluated_events,
            "evidence_citations": retrieval_res["evidence"]
        }
