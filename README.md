# 🔬 Autonomous AI Research Synthesizer

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange)
![Gemini](https://img.shields.io/badge/Google-Gemini_API-blue?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Cache-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)

An enterprise-grade, multi-agent AI research engine that autonomously scouts academic literature (arXiv), critically evaluates methodologies, and synthesizes comprehensive markdown reports. Designed for production, this architecture features **semantic vector caching** to optimize API costs and **autonomous fault tolerance** to handle rate limits and network degradation gracefully.

## 🏗️ Architecture & Key Features

*   **Multi-Agent Orchestration:** Utilizes CrewAI to manage a specialized pipeline of three distinct agents:
    *   *Academic Literature Scout:* Interrogates the arXiv API using optimized boolean queries to extract relevant abstracts.
    *   *Methodology Critic:* Evaluates sample sizes, dataset biases, and logical gaps, assigning scientific confidence scores.
    *   *Synthesis Writer:* Compiles the raw data and critiques into a cohesive, professional markdown literature review.
*   **Semantic Vector Caching (ChromaDB):** Converts user hypotheses into vector embeddings (via Google `models/embedding-001`). Semantically identical queries bypass the LLM entirely, returning cached reports to eliminate redundant compute costs.
*   **Autonomous Fault Tolerance:** Equipped with internal vector memory, allowing agents to fall back on previously embedded abstracts if external APIs (like arXiv) timeout or trigger `HTTP 429` rate limits.
*   **Interactive Web UI:** A responsive frontend built with Streamlit featuring live telemetry badges indicating whether data was generated fresh or served from the local cache.

---

## ⚡ Performance Telemetry: The Power of Semantic Caching

To demonstrate the cost-efficiency of this architecture, below is an execution trace of a real-world prompt, highlighting the resource difference between a cold start and a semantic cache hit.

**Target Hypothesis:** *"The impact of Retrieval-Augmented Generation (RAG) on reducing hallucination rates in medical diagnostics."*

| Metric | 🔍 Run 1: Cold Start (Cache Miss) | ⚡ Run 2: Semantic Cache Hit |
| :--- | :--- | :--- |
| **Execution Path** | Full Multi-Agent Graph (Scout → Critic → Writer) | Intercepted by ChromaDB Vector Search |
| **arXiv API Calls** | 1 (Live Data Fetch) | 0 (Bypassed) |
| **LLM Inference** | Active (Gemini API) | Bypassed |
| **Token Consumption**| ~12,000 - 15,000 Tokens | **0 Tokens** |
| **System Latency** | ~90 - 120 seconds | **< 0.1 seconds** |
| **Fault Tolerance** | Resilient to 429 Timeouts via local memory fallback | Immune to network outages |

*Note: On Run 2, the user can input a rephrased query such as "Does using RAG decrease hallucinations for AI in healthcare?" The embedding distance algorithm (threshold < 0.25) accurately maps the semantic intent to the original query, triggering the zero-token cache hit.*

---

## 🛠️ Technology Stack

*   **Backend:** Python 3.13, CrewAI (Agent Orchestration)
*   **LLM Provider:** Google Gemini API 
*   **Embeddings & Cache:** ChromaDB, `google-generativeai` (Google Embedding Models)
*   **Frontend:** Streamlit
*   **Tooling:** `crewai_tools` (ArxivPaperTool), `uv` (Package Management)

---

## 🚀 Installation & Setup

**1. Clone the repository and install dependencies using `uv`:**
```bash
git clone [https://github.com/yourusername/autonomous-research-synthesizer.git](https://github.com/yourusername/autonomous-research-synthesizer.git)
cd autonomous-research-synthesizer
uv venv
source .venv/bin/activate
uv add crewai crewai-tools streamlit chromadb google-generativeai python-dotenv