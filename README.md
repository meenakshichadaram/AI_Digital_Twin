# AI Digital Twin of Knowledge: Personalized Learning Companion

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/Python-3.12%2B-green.svg)
![Framework](https://img.shields.io/badge/LangGraph-0.1.1-purple.svg)
![Framework](https://img.shields.io/badge/LangChain-0.2%2B-blue.svg)
![UI](https://img.shields.io/badge/Streamlit-Glassmorphic-FF4B4B.svg)

> **A Production-Grade SaaS AI Learning Companion** that continuously profiles the learner over time, maintaining short and long-term memory, retrieving course documents via ChromaDB RAG, calling 15 specialized pedagogical tools, and dynamically routing workflows using a **LangGraph** state machine.

---

## 🌟 Key Capabilities & Architectural Pillars

### 1. 🧠 Dynamic Dual Memory System (SQLite & LLM Extraction)
- **Short-Term Memory**: Conversation window tracking recent chat history and transient state.
- **Long-Term Profile Persistence**: Persistent tracking of learner demographics, weak/strong concepts, exam dates, study hours, learning style, and quiz performance stored in SQLite.
- **Self-Improving Memory Pipeline**: After every interaction, an automated memory agent extracts key facts (e.g. *"I am weak in Convolutional Neural Networks"*) and updates the database seamlessly.

### 2. 📚 Vector Retrieval-Augmented Generation (ChromaDB RAG)
- Multi-format document ingestion: **PDF**, **DOCX**, **TXT**, **PPTX**.
- Smart document chunking via `RecursiveCharacterTextSplitter`.
- Local open-source embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Exact citation metadata extraction: returns file name, page/section number, and exact text chunks.

### 3. 🛠️ 15 Production-Grade Pedagogical Tools
1. **Quiz Generator**: Generates customized multiple-choice tests with detailed explanations.
2. **Flashcard Generator**: Creates question/answer study cards for active recall.
3. **Revision Planner**: Formulates daily and weekly study schedules leading up to exams.
4. **Weak Topic Analyzer**: Evaluates quiz performance and detects target knowledge gaps.
5. **Learning Path Generator**: Builds tailored roadmaps (Beginner to Advanced) with projects.
6. **Study Time Calculator**: Estimates total hours needed based on topic complexity.
7. **PDF Summarizer**: Condenses dense study materials into core takeaways.
8. **Notes Generator**: Produces structured markdown study notes.
9. **Mind Map Generator**: Outputs structured hierarchical topic maps.
10. **Concept Explainer**: Explains complex concepts using intuitive analogies.
11. **Daily Goal Generator**: Creates actionable, realistic daily study tasks.
12. **Motivation Tool**: Delivers personalized encouragement tailored to current progress.
13. **Resource Recommendation Tool**: Curates YouTube videos, books, and practice sites.
14. **Progress Tracker**: Logs completed milestones and concepts.
15. **Exam Readiness Calculator**: Computes a percentage score of exam preparedness.

### 4. 🔀 Orchestrated LangGraph Workflow Engine
StateGraph pipeline featuring deterministic intent routing, memory enrichment, RAG checks, tool execution, and memory auto-sync:

```
[Start]
  │
  ▼
[Intent Classifier Node]
  │
  ├───────────────────────┬────────────────────────┐
  ▼                       ▼                        ▼
[Memory Retrieval Node] [RAG Retrieval Node] [Tool Execution Node]
  │                       │                        │
  └───────────────────────┼────────────────────────┘
                          ▼
                    [LLM Generation Node]
                          │
                          ▼
                 [Memory Auto-Update Node]
                          │
                          ▼
                        [End]
```

---

## 🎨 UI / UX Aesthetics & Interface

Designed with inspiration from modern SaaS products (OpenAI, Notion AI, Vercel, Linear, Stripe):
- **Glassmorphism**: Translucent card layouts with subtle backdrops and CSS blur effects.
- **Theme Modes**: Dark Mode & Light Mode support.
- **Dynamic Analytics**: Plotly interactive charts for study hours, weak topics, and readiness scores.
- **Multi-Page Navigation**:
  - `Landing Page`: Hero presentation, feature highlights, quick start.
  - `AI Chat`: Streamed responses, markdown rendering, tool output displays.
  - `Memory Dashboard`: Profile inspector, memory timeline, editable knowledge graph.
  - `Knowledge Base`: Document uploader, vector store status, chunk visualizer.
  - `Learning Analytics`: Plotly performance charts & memory growth analytics.
  - `Quiz Center`: Interactive quiz solver with score tracking.
  - `Revision Planner`: Drag-and-drop daily study planner.
  - `Flashcards`: Flip card deck for active recall.
  - `Settings`: LLM selection (OpenAI / Gemini / Groq), hyperparameters, data export.

---

## 📁 Repository Directory Structure

```
AI_Digital_Twin/
├── app.py                     # Main Streamlit SaaS application entry point
├── graph.py                   # LangGraph StateGraph engine & node definitions
├── state.py                   # Pydantic & TypedDict state definitions
├── memory.py                  # Dual Memory System & SQLite manager
├── rag.py                     # Vector database processing & retrieval pipeline
├── tools.py                   # 15 Custom Pedagogical Tools implementation
├── prompts.py                 # System prompts, persona definitions, instructions
├── config.py                  # Global settings, paths, environment configuration
├── database.py                # SQLite schema definition & raw SQL helpers
├── requirements.txt           # Pinned python dependencies
├── README.md                  # Project documentation & execution guide
├── .env.example               # Environment variables template
├── utils/                     # Formatting, export (PDF/MD), & math helpers
├── agents/                    # Intent classifier, memory extractor, agent nodes
├── memory/                    # SQLite database persistence location
├── vectorstore/               # ChromaDB index persistence directory
├── uploaded_docs/             # Cached user uploaded study materials
├── assets/                    # Static UI media assets
│   └── css/                   # Glassmorphic custom CSS styling rules
├── components/                # Modular Streamlit UI widgets & visualizers
└── pages/                     # Multi-page views for Streamlit layout
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- **Python 3.12+**
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/AI_Digital_Twin.git
cd AI_Digital_Twin

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your API key(s):
```bash
cp .env.example .env
```
Edit `.env`:
```env
LLM_PROVIDER=openai  # Options: openai, google, groq
OPENAI_API_KEY=sk-proj-your-key-here
```

### 5. Launch the SaaS Platform
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Testing

Run unit & integration tests across modules:
```bash
python -m py_compile config.py
python -c "import config; print(config.settings.validate())"
```

---

## 🛡️ Security & Privacy
- Zero hardcoded API keys; strictly enforced via environment variables.
- User data & document chunks remain isolated on local disk in SQLite & ChromaDB.

---

## 📜 License
Released under the [MIT License](LICENSE).
