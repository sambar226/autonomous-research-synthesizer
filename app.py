import streamlit as st
from research_engine import run_research

st.set_page_config(page_title="Academic Research Engine", page_icon="🔬", layout="centered")

st.title("🔬 Autonomous Research Synthesizer")
st.markdown("Enter a hypothesis below. The Multi-Agent Crew will scout arXiv, critique methodologies, and synthesize a literature review.")

# Input Field
hypothesis = st.text_area("Research Hypothesis:", placeholder="e.g., The impact of RAG on reducing hallucination rates...")

# Run Button
if st.button("Synthesize Research", type="primary"):
    if not hypothesis:
        st.warning("Please enter a hypothesis first.")
    else:
        with st.spinner("Processing request..."):
            # Unpack both the report and the cache flag from the backend
            report, is_cached = run_research(hypothesis)
            
        # Display the custom UI indicator based on the flag
        if is_cached:
            st.toast("Retrieved instantly from Semantic Cache!", icon="⚡")
            st.success("🟢 **Source Indicator:** Served from local vector cache. (API Cost: 0 tokens)")
        else:
            st.toast("Generated fresh by CrewAI Agents!", icon="🤖")
            st.warning("🟠 **Source Indicator:** Generated via live Gemini API and arXiv search.")
            
        st.markdown("---")
        st.markdown(report)