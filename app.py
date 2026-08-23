import streamlit as st
import pandas as pd
from datetime import datetime

from models import (
    CompanyProfile, CompanyProduct, PriorityLevel, Event, Entity, EntityType, EventType, EventSource
)
from ingestion.connectors import IngestionManager
from ingestion.extraction import ExtractionPipeline
from knowledge_graph import KnowledgeGraph
from retrieval import HybridRetriever
from agents.competitor_agent import CompetitorIntelligenceAgent
from agents.daily_agent import DailyIntelligenceAgent
from signal_fusion import SignalFusionEngine

# Page Config
st.set_page_config(
    page_title="Biopharma Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)

# Initialize Session State
if "company_profile" not in st.session_state:
    st.session_state.company_profile = CompanyProfile(
        id="profile_101",
        company_name="Apex Therapeutics",
        profile_description="Clinical-stage biopharmaceutical company focusing on novel targeted oncology and kinase inhibitors.",
        therapeutic_areas=["Oncology", "Non-Small Cell Lung Cancer"],
        geographic_markets=["USA", "EU", "Japan"],
        products=[
            CompanyProduct(
                id="prod_1",
                drug_name="Apex-701",
                target="EGFR",
                indication="Non-Small Cell Lung Cancer (NSCLC)",
                development_stage="Phase 2",
                geographic_markets=["USA", "EU"]
            )
        ],
        known_competitors=["OncoX Bio", "NeuroGen Therapeutics", "AstraZeneca"],
        strategic_objectives=[
            "Prepare Phase 3 global trial protocol",
            "Monitor European competitor regulatory filings (EMA)",
            "Discover licensing partners for EU commercial distribution"
        ]
    )

if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = KnowledgeGraph()

if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever(st.session_state.knowledge_graph)

if "ingested_events" not in st.session_state:
    st.session_state.ingested_events = []

if "daily_agent" not in st.session_state:
    st.session_state.daily_agent = DailyIntelligenceAgent()

# Helper to load initial data
def load_sample_data():
    if not st.session_state.ingested_events:
        mgr = IngestionManager()
        pipeline = ExtractionPipeline()
        raw_items = mgr.fetch_all()
        events = pipeline.process_batch(raw_items)
        st.session_state.ingested_events = events
        st.session_state.retriever.index_events(events)

load_sample_data()

# App Header
st.title("🧬 Biopharma Strategic Intelligence Platform")
st.markdown("*Continuously monitoring biopharma landscapes, connecting signals, and delivering evidence-backed intelligence.*")

st.sidebar.title("🏢 Company Context")
st.sidebar.subheader(st.session_state.company_profile.company_name)
st.sidebar.caption(f"Therapeutic Areas: {', '.join(st.session_state.company_profile.therapeutic_areas)}")
st.sidebar.caption(f"Active Markets: {', '.join(st.session_state.company_profile.geographic_markets)}")
st.sidebar.caption(f"Pipeline Lead: {st.session_state.company_profile.products[0].drug_name} ({st.session_state.company_profile.products[0].target})")

# Navigation Tabs
tab_feed, tab_competitor, tab_regulatory, tab_research, tab_profile = st.tabs([
    "🔴 Prioritized Intelligence Feed",
    "⚔️ Competitor Intelligence",
    "📜 Regulatory Impact View",
    "🔬 Research & Signal Fusion",
    "⚙️ Company Profile Onboarding"
])

# -----------------------------------------------------------------------------
# TAB 1: PRIORITIZED INTELLIGENCE FEED
# -----------------------------------------------------------------------------
with tab_feed:
    st.header("⚡ Live Intelligence & Alert Feed")
    st.caption("Answers: 'What changed in the biopharma landscape that matters to my company, why does it matter, and what deserves my attention?'")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("🔄 Refresh Pipeline Data"):
            load_sample_data()
            st.success("Ingestion pipeline synchronized!")

    alerts = st.session_state.daily_agent.process_incoming_events(
        st.session_state.ingested_events,
        st.session_state.company_profile
    )

    if not alerts:
        st.info("No urgent alerts detected for current profile context.")
    else:
        for alert in alerts:
            priority_color = "🔴" if alert.priority == PriorityLevel.HIGH else ("🟡" if alert.priority == PriorityLevel.MEDIUM else "🟢")

            with st.expander(f"{priority_color} {alert.title}", expanded=(alert.priority == PriorityLevel.HIGH)):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**Summary:** {alert.summary}")
                    st.markdown(f"**Recommended Strategic Action:** `{alert.recommended_action}`")
                with c2:
                    if alert.impact_score:
                        st.metric("Relevance Score", f"{alert.impact_score.relevance}%")
                        st.metric("Potential Impact", f"{alert.impact_score.impact}%")
                        st.metric("Urgency", f"{alert.impact_score.urgency}%")
                        st.caption(f"Confidence: {alert.impact_score.confidence}%")

                if alert.event:
                    st.divider()
                    st.caption(f"Source: {alert.event.source.value} | Jurisdiction: {alert.event.jurisdiction} | Date: {alert.event.timestamp.strftime('%Y-%m-%d')}")
                    if alert.event.source_url:
                        st.markdown(f"[🔗 Direct Evidence Source Link]({alert.event.source_url})")

# -----------------------------------------------------------------------------
# TAB 2: COMPETITOR INTELLIGENCE
# -----------------------------------------------------------------------------
with tab_competitor:
    st.header("⚔️ Competitor Intelligence Agent")
    st.caption("Deep-dive tracking of competitor timelines, trials, filings, and strategic partnerships.")

    selected_competitor = st.selectbox(
        "Select Competitor / Drug Target to Analyze",
        options=st.session_state.company_profile.known_competitors + ["OncoX-201", "Neuro-88"]
    )

    if st.button("Run Competitor Intelligence Analysis"):
        agent = CompetitorIntelligenceAgent(
            st.session_state.knowledge_graph,
            st.session_state.retriever
        )
        
        with st.spinner(f"Analyzing multi-source intelligence for {selected_competitor}..."):
            analysis = agent.analyze_competitor(selected_competitor, st.session_state.company_profile)

        st.success("Analysis Complete")
        st.info(analysis["strategic_summary"])

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("📅 Competitor Event Timeline")
            if analysis["timeline"]:
                for item in analysis["timeline"]:
                    st.markdown(f"• **{item.get('timestamp', '')[:10]}** - `{item.get('event_type')}`: **{item.get('label')}**")
                    st.caption(item.get('summary', ''))
            else:
                st.write("No historical timeline events found.")

        with col_t2:
            st.subheader("⚡ Cross-Source Fused Signals")
            if analysis["fused_signals"]:
                for sig in analysis["fused_signals"]:
                    st.markdown(f"**{sig.title}**")
                    st.markdown("*Observed Facts:*")
                    for fact in sig.observed_facts:
                        st.caption(f"- {fact}")
                    st.markdown("*Model-Generated Inferences:*")
                    for inf in sig.model_generated_inferences:
                        st.caption(f"- 💡 {inf}")
            else:
                st.write("No multi-signal patterns detected yet.")

# -----------------------------------------------------------------------------
# TAB 3: REGULATORY IMPACT VIEW
# -----------------------------------------------------------------------------
with tab_regulatory:
    st.header("📜 Regulatory Impact Intelligence")
    st.caption("Applicability depends on: authority + jurisdiction + product + indication + development stage + market")

    reg_events = [e for e in st.session_state.ingested_events if e.source in [EventSource.REGULATORY_BODY, EventSource.SEC_FILINGS]]
    
    if not reg_events:
        st.info("No regulatory events currently indexed.")
    else:
        for evt in reg_events:
            st.subheader(f"🏛️ {evt.title}")
            st.write(evt.summary)
            
            c_reg1, c_reg2, c_reg3 = st.columns(3)
            with c_reg1:
                st.markdown(f"**Jurisdiction / Authority:** {evt.jurisdiction}")
            with c_reg2:
                st.markdown(f"**Source:** {evt.source.value}")
            with c_reg3:
                st.markdown(f"**Affected Market Match:** {'✅ Matching EU/US Market' if any(m in (evt.jurisdiction or '') for m in st.session_state.company_profile.geographic_markets) else '⚠️ Outside Active Market'}")

# -----------------------------------------------------------------------------
# TAB 4: RESEARCH & SIGNAL FUSION
# -----------------------------------------------------------------------------
with tab_research:
    st.header("🔬 Hybrid Search & Multi-Signal Fusion")
    st.caption("Combining Vector Semantic Search and Knowledge Graph multi-hop retrieval")

    query = st.text_input("Search biopharma scientific literature, targets, or trials:", value="EGFR NSCLC")

    if query:
        ret_res = st.session_state.retriever.hybrid_retrieve(query)

        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.subheader("📄 Semantic Vector Evidence")
            for evt in ret_res["vector_events"]:
                st.markdown(f"**[{evt.source.value}] {evt.title}**")
                st.caption(evt.summary)
                st.divider()

        with c_res2:
            st.subheader("🕸️ Knowledge Graph Relational Multi-Hop Traversals")
            if ret_res["graph_paths"]:
                for g_item in ret_res["graph_paths"]:
                    node = g_item["node"]
                    st.markdown(f"**Node:** `{node.get('label')}` ({node.get('entity_type', node.get('type'))})")
                    st.caption(f"Path: {' '.join(g_item['path'])}")
            else:
                st.write("No multi-hop relational nodes returned for query.")

# -----------------------------------------------------------------------------
# TAB 5: COMPANY PROFILE ONBOARDING
# -----------------------------------------------------------------------------
with tab_profile:
    st.header("⚙️ Company Intelligence Profile Onboarding")
    st.caption("Configure company profile context used for relevance and impact scoring.")

    with st.form("profile_form"):
        c_name = st.text_input("Company Name", value=st.session_state.company_profile.company_name)
        desc = st.text_area("Profile Description", value=st.session_state.company_profile.profile_description)
        tas = st.text_input("Therapeutic Areas (comma separated)", value=", ".join(st.session_state.company_profile.therapeutic_areas))
        mkts = st.text_input("Geographic Markets (comma separated)", value=", ".join(st.session_state.company_profile.geographic_markets))
        comps = st.text_input("Known Competitors (comma separated)", value=", ".join(st.session_state.company_profile.known_competitors))

        submitted = st.form_submit_button("Save Company Profile")
        if submitted:
            st.session_state.company_profile.company_name = c_name
            st.session_state.company_profile.profile_description = desc
            st.session_state.company_profile.therapeutic_areas = [x.strip() for x in tas.split(",") if x.strip()]
            st.session_state.company_profile.geographic_markets = [x.strip() for x in mkts.split(",") if x.strip()]
            st.session_state.company_profile.known_competitors = [x.strip() for x in comps.split(",") if x.strip()]
            st.success("Company Profile updated successfully!")
