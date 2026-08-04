"""
AI Digital Twin of Knowledge - LangGraph Orchestration Engine

Defines the complete LangGraph StateGraph flow:
Intent Classifier -> Memory Retrieval -> RAG Retrieval -> Tool Execution -> LLM Synthesis -> Memory Auto-Sync.
"""

from typing import Dict, Any, List, Optional
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    END = "END"

from state import GraphState, IntentCategory
from memory import memory_manager
from rag import rag_pipeline
from tools import execute_tool, TOOL_REGISTRY
from agents.intent_classifier import intent_classifier
from agents.llm_factory import LLMFactory
from prompts import SYSTEM_PERSONA_PROMPT, LLM_SYNTHESIS_PROMPT


# ==================== NODE DEFINITIONS ====================

def intent_classifier_node(state: GraphState) -> GraphState:
    """Classify user query intent and topic."""
    user_query = state.get("user_query", "")
    res = intent_classifier.classify_intent(user_query)

    state["intent"] = res["intent"]
    state["intent_confidence"] = res["confidence"]
    if "tool_args" not in state or not state["tool_args"]:
        state["tool_args"] = {"topic": res.get("topic", user_query)}

    return state


def memory_retrieval_node(state: GraphState) -> GraphState:
    """Retrieve learner profile and long-term facts from SQLite memory."""
    user_id = state.get("user_id", "default_user")
    profile = memory_manager.get_learner_profile(user_id)
    chat_history = memory_manager.get_short_term_memory(user_id, limit=10)

    state["learner_profile"] = profile
    state["chat_history"] = chat_history
    return state


def rag_retrieval_node(state: GraphState) -> GraphState:
    """Query ChromaDB vector database for relevant study document chunks."""
    user_query = state.get("user_query", "")
    user_id = state.get("user_id", "default_user")

    chunks = rag_pipeline.query_relevant_chunks(query=user_query, user_id=user_id, top_k=4)
    state["retrieved_docs"] = chunks
    return state


def tool_execution_node(state: GraphState) -> GraphState:
    """Dispatch and execute target pedagogical tool."""
    intent = state.get("intent", "")
    user_query = state.get("user_query", "")
    user_id = state.get("user_id", "default_user")
    tool_args = dict(state.get("tool_args") or {})
    tool_args["user_query"] = user_query
    if "topic" not in tool_args or not tool_args["topic"]:
        tool_args["topic"] = user_query

    intent_tool_map = {
        IntentCategory.QUIZ_REQUEST.value: "generate_quiz",
        IntentCategory.FLASHCARDS.value: "generate_flashcards",
        IntentCategory.REVISION_PLAN.value: "create_revision_planner",
        IntentCategory.WEAK_TOPIC_ANALYSIS.value: "analyze_weak_topics",
        IntentCategory.LEARNING_PATH.value: "generate_learning_path",
        IntentCategory.STUDY_TIME.value: "calculate_study_time",
        IntentCategory.SUMMARIZATION.value: "summarize_pdf_notes",
        IntentCategory.EXPLAIN_CONCEPT.value: "explain_concept",
        IntentCategory.MIND_MAP.value: "generate_mind_map",
        IntentCategory.DAILY_GOAL.value: "generate_daily_goals",
        IntentCategory.MOTIVATION.value: "generate_motivation",
        IntentCategory.RESOURCE_RECOMMENDATION.value: "recommend_resources",
        IntentCategory.PROGRESS_TRACKING.value: "track_progress",
        IntentCategory.EXAM_READINESS.value: "calculate_exam_readiness",
    }

    target_tool = intent_tool_map.get(intent)
    if target_tool and target_tool in TOOL_REGISTRY:
        state["active_tool"] = target_tool
        result = execute_tool(tool_name=target_tool, tool_args=tool_args, user_id=user_id)
        state["tool_result"] = result
    else:
        state["active_tool"] = None
        state["tool_result"] = None

    return state


def llm_generation_node(state: GraphState) -> GraphState:
    """Synthesize final response using LLM or Fallback engine."""
    user_query = state.get("user_query", "")
    user_id = state.get("user_id", "default_user")
    retrieved_docs = state.get("retrieved_docs", [])
    tool_result = state.get("tool_result")

    mem_context = memory_manager.get_formatted_memory_context(user_id)
    persona = SYSTEM_PERSONA_PROMPT.format(memory_context=mem_context)

    rag_context = rag_pipeline.format_rag_context(retrieved_docs) if retrieved_docs else ""
    tool_output_context = tool_result.get("rendered_output", "") if tool_result else ""

    prompt = LLM_SYNTHESIS_PROMPT.format(
        system_persona=persona,
        rag_context=rag_context,
        tool_output_context=tool_output_context,
        user_query=user_query
    )

    llm = LLMFactory.get_llm(temperature=0.7)
    try:
        response_obj = llm.invoke(prompt)
        response_text = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
    except Exception:
        response_text = tool_output_context if tool_output_context else f"Response for **{user_query}**."

    if tool_result and tool_result.get("rendered_output") and tool_result["rendered_output"] not in response_text:
        response_text = f"{tool_result['rendered_output']}\n\n---\n{response_text}"

    if retrieved_docs:
        citations_md = rag_pipeline.format_citations_markdown(retrieved_docs)
        if citations_md and citations_md not in response_text:
            response_text += citations_md
            state["citations"] = retrieved_docs

    state["generated_response"] = response_text
    return state


def memory_update_node(state: GraphState) -> GraphState:
    """Automatically extract personal facts from query and persist in SQLite memory."""
    user_id = state.get("user_id", "default_user")
    user_query = state.get("user_query", "")
    generated_response = state.get("generated_response", "")
    citations = state.get("citations", [])

    memory_manager.add_chat_turn(
        user_id=user_id,
        role="user",
        content=user_query
    )
    memory_manager.add_chat_turn(
        user_id=user_id,
        role="assistant",
        content=generated_response,
        citations=citations
    )

    updates = memory_manager.auto_extract_and_update_memory(
        user_id=user_id,
        user_message=user_query,
        assistant_response=generated_response
    )
    state["memory_updates"] = updates
    return state


# ==================== STATE GRAPH CONSTRUCTION ====================

def route_by_intent(state: GraphState) -> str:
    """Conditional routing edge function."""
    intent = state.get("intent", "")

    tool_intents = [
        IntentCategory.QUIZ_REQUEST.value, IntentCategory.FLASHCARDS.value,
        IntentCategory.REVISION_PLAN.value, IntentCategory.WEAK_TOPIC_ANALYSIS.value,
        IntentCategory.LEARNING_PATH.value, IntentCategory.STUDY_TIME.value,
        IntentCategory.SUMMARIZATION.value, IntentCategory.EXPLAIN_CONCEPT.value,
        IntentCategory.MIND_MAP.value, IntentCategory.DAILY_GOAL.value,
        IntentCategory.MOTIVATION.value, IntentCategory.RESOURCE_RECOMMENDATION.value,
        IntentCategory.PROGRESS_TRACKING.value, IntentCategory.EXAM_READINESS.value
    ]

    if intent in tool_intents:
        return "tool_node"
    elif intent in [IntentCategory.RAG_QUERY.value, IntentCategory.SUMMARIZATION.value]:
        return "rag_node"
    return "memory_node"


def build_digital_twin_graph():
    """Assemble and compile the LangGraph workflow StateGraph."""
    if HAS_LANGGRAPH:
        workflow = StateGraph(GraphState)

        workflow.add_node("intent_classifier", intent_classifier_node)
        workflow.add_node("memory_retrieval", memory_retrieval_node)
        workflow.add_node("rag_retrieval", rag_retrieval_node)
        workflow.add_node("tool_execution", tool_execution_node)
        workflow.add_node("llm_generation", llm_generation_node)
        workflow.add_node("memory_update", memory_update_node)

        workflow.set_entry_point("intent_classifier")

        workflow.add_conditional_edges(
            "intent_classifier",
            route_by_intent,
            {
                "tool_node": "tool_execution",
                "rag_node": "rag_retrieval",
                "memory_node": "memory_retrieval"
            }
        )

        workflow.add_edge("tool_execution", "llm_generation")
        workflow.add_edge("rag_retrieval", "llm_generation")
        workflow.add_edge("memory_retrieval", "llm_generation")

        workflow.add_edge("llm_generation", "memory_update")
        workflow.add_edge("memory_update", END)

        return workflow.compile()
    else:
        return None


# Compiled Graph Instance
digital_twin_graph = build_digital_twin_graph()


def run_digital_twin_workflow(user_query: str, user_id: str = "default_user") -> Dict[str, Any]:
    """Execute complete LangGraph workflow for user query."""
    state: GraphState = {
        "user_id": user_id,
        "user_query": user_query,
        "intent": "",
        "intent_confidence": 0.0,
        "chat_history": [],
        "learner_profile": {},
        "retrieved_docs": [],
        "active_tool": None,
        "tool_args": {},
        "tool_result": None,
        "generated_response": "",
        "citations": [],
        "memory_updates": {},
        "error": None
    }

    if digital_twin_graph:
        return digital_twin_graph.invoke(state)

    # Native Pipeline Fallback (Mirroring LangGraph Node Sequence Exactly)
    state = intent_classifier_node(state)
    target = route_by_intent(state)
    if target == "tool_node":
        state = tool_execution_node(state)
    elif target == "rag_node":
        state = rag_retrieval_node(state)
    else:
        state = memory_retrieval_node(state)

    state = llm_generation_node(state)
    state = memory_update_node(state)
    return state
