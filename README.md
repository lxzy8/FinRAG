# AI-Powered Biopharma Intelligence Platform

An enterprise AI platform designed for pharmaceutical and biotechnology companies to continuously monitor scientific, clinical, regulatory, patent, and corporate information across multi-source landscapes.

## Key Features

- **Company Intelligence Profile Onboarding**: Customizes intelligence tracking based on company portfolio, drug targets, indications, stage, active geographic markets, and strategic objectives.
- **Multi-Source Ingestion Engine**: Connects to ClinicalTrials.gov, PubMed, SEC Form 8-K/10-K filings, and corporate press releases.
- **Temporal Knowledge Graph**: Maintains historical entity/event state and directional relationships (`Company -> develops -> Drug -> targets -> Target -> treats -> Disease`).
- **Hybrid Retrieval**: Combines semantic vector retrieval with graph multi-hop relational retrieval for evidence synthesis.
- **Cross-Source Signal Fusion**: Connects weak signals across clinical, filing, and partnership sources into strategic insights with explicit distinction between directly observed facts, evidence-supported interpretations, and AI model inferences.
- **Impact Engine**: Calculates Relevance, Impact, Urgency, and Confidence scores with evidence-backed explanations.
- **Competitor Intelligence Agent**: Automatically discovers competitor timelines, trial updates, and strategic moves.
- **Daily Monitoring Agent**: Answers "What changed?", "Why does it matter?", and delivers prioritized alerts (🔴 High, 🟡 Medium, 🟢 Low).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```

3. Run Tests:
   ```bash
   pytest tests/
   ```
