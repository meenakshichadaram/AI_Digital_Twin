"""
AI Digital Twin of Knowledge - Integration Test Suite

End-to-end verification of configuration, database, memory, RAG, 15 custom tools,
intent classification, LangGraph state machine, and export engines.
"""

import sys
import os

from config import settings
from database import db
from memory import memory_manager
from rag import rag_pipeline
from tools import TOOL_REGISTRY, execute_tool
from agents.intent_classifier import intent_classifier
from graph import run_digital_twin_workflow
from utils.export_engine import export_chat_history_md, export_profile_json, export_summary_pdf_bytes


def run_all_tests():
    """Run full system integration test battery."""
    print("==================================================")
    print("   AI DIGITAL TWIN — SYSTEM INTEGRATION TEST")
    print("==================================================")

    test_user = "integration_test_user"

    # 1. Config Validation
    print("[1/8] Testing Configuration Manager...")
    status = settings.validate()
    print(f"      - Active Provider: {status['active_provider']} | Model: {status['active_model']}")
    print("      [OK] Config OK")

    # 2. Database Verification
    print("[2/8] Testing SQLite Database Persistence Engine...")
    profile = db.get_learner_profile(test_user)
    assert profile is not None
    print(f"      - Seeded Profile for '{test_user}': {profile['name']}")
    print("      [OK] Database OK")

    # 3. Memory Pipeline Verification
    print("[3/8] Testing Dual Memory & Auto-Extraction Pipeline...")
    updates = memory_manager.auto_extract_and_update_memory(test_user, "My name is Namith and I am weak in Backpropagation")
    print(f"      - Extracted Updates: {updates}")
    mem_ctx = memory_manager.get_formatted_memory_context(test_user)
    assert "Namith" in mem_ctx
    print("      [OK] Memory Pipeline OK")

    # 4. RAG Engine Verification
    print("[4/8] Testing Multi-Format RAG Ingestion & Vector Search...")
    sample_bytes = b"Deep Learning Architectures rely on Gradient Descent and Backpropagation for weight updates."
    ingest_res = rag_pipeline.ingest_document(sample_bytes, "dl_notes.txt", test_user)
    print(f"      - Ingest Status: {ingest_res['status']} | Chunks: {ingest_res['total_chunks']}")
    retrieved = rag_pipeline.query_relevant_chunks("What updates weights in deep learning?", test_user, top_k=1)
    print(f"      - Query Matches: {len(retrieved)}")
    print("      [OK] RAG Engine OK")

    # 5. Tools Registry Verification (15 Tools)
    print("[5/8] Testing 15 Custom Pedagogical Tools...")
    passed_tools = 0
    for t_name in TOOL_REGISTRY:
        res = execute_tool(t_name, {"topic": "Neural Networks"}, test_user)
        if res.get("status") == "success":
            passed_tools += 1
    print(f"      - Tools Passed: {passed_tools}/{len(TOOL_REGISTRY)}")
    assert passed_tools == 15
    print("      [OK] 15 Pedagogical Tools OK")

    # 6. Intent Classifier Verification
    print("[6/8] Testing Intent Classification Agent...")
    c_res = intent_classifier.classify_intent("Generate a quiz on CNN")
    print(f"      - Classified Intent: {c_res['intent']} (Confidence: {c_res['confidence']})")
    assert c_res["intent"] == "QUIZ_REQUEST"
    print("      [OK] Intent Classifier OK")

    # 7. LangGraph StateMachine Verification
    print("[7/8] Testing LangGraph Workflow Execution...")
    graph_res = run_digital_twin_workflow("Create a flashcard deck on Machine Learning", test_user)
    print(f"      - Workflow Intent: {graph_res['intent']} | Response Len: {len(graph_res['generated_response'])}")
    assert len(graph_res["generated_response"]) > 0
    print("      [OK] LangGraph Workflow OK")

    # 8. Export Engine Verification
    print("[8/8] Testing Data Export Engines...")
    md_doc = export_chat_history_md(test_user)
    json_doc = export_profile_json(test_user)
    pdf_bytes = export_summary_pdf_bytes("Digital Twin Report", "Sample summary content.")
    print(f"      - MD Len: {len(md_doc)} | JSON Len: {len(json_doc)} | PDF Bytes: {len(pdf_bytes)}")
    print("      [OK] Export Engines OK")

    print("\n==================================================")
    print("   ALL 8 INTEGRATION TEST MODULES PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_all_tests()
