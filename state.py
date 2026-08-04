"""
AI Digital Twin of Knowledge - State Management

Defines data schemas, Pydantic validation models (with fallback), and LangGraph
TypedDict state definitions used across agent nodes, memory systems, RAG, and tools.
"""

from typing import List, Dict, Any, Optional, Union, TypedDict
from enum import Enum
from datetime import datetime

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return self.__dict__
    def Field(default=None, default_factory=None, **kwargs):
        if default_factory:
            return default_factory()
        return default


class IntentCategory(str, Enum):
    """Supported intent classification categories."""
    GENERAL_CHAT = "GENERAL_CHAT"
    MEMORY_QUERY = "MEMORY_QUERY"
    QUIZ_REQUEST = "QUIZ_REQUEST"
    FLASHCARDS = "FLASHCARDS"
    REVISION_PLAN = "REVISION_PLAN"
    EXPLAIN_CONCEPT = "EXPLAIN_CONCEPT"
    RAG_QUERY = "RAG_QUERY"
    SUMMARIZATION = "SUMMARIZATION"
    LEARNING_PATH = "LEARNING_PATH"
    DAILY_GOAL = "DAILY_GOAL"
    WEAK_TOPIC_ANALYSIS = "WEAK_TOPIC_ANALYSIS"
    STUDY_TIME = "STUDY_TIME"
    MOTIVATION = "MOTIVATION"
    PROGRESS_TRACKING = "PROGRESS_TRACKING"
    EXAM_READINESS = "EXAM_READINESS"
    MIND_MAP = "MIND_MAP"
    RESOURCE_RECOMMENDATION = "RESOURCE_RECOMMENDATION"


if HAS_PYDANTIC:
    class LearnerProfile(BaseModel):
        """Pydantic model representing comprehensive learner profile and state."""
        name: str = Field(default="Learner", description="User's full name")
        age: Optional[int] = Field(default=None, description="Age in years")
        college: Optional[str] = Field(default=None, description="College or Institution")
        university: Optional[str] = Field(default=None, description="University name")
        branch: Optional[str] = Field(default=None, description="Field of study or major")
        subjects: List[str] = Field(default_factory=list, description="Enrolled subjects")
        learning_goals: List[str] = Field(default_factory=list, description="Short and long term goals")
        exam_dates: Dict[str, str] = Field(default_factory=dict, description="Subject -> Exam Date mapping")
        weak_concepts: List[str] = Field(default_factory=list, description="Detected or declared weak topics")
        strong_concepts: List[str] = Field(default_factory=list, description="Mastered topics")
        preferred_learning_style: str = Field(default="Visual & Practical", description="Learning style preference")
        daily_study_hours: float = Field(default=3.0, description="Target daily study hours")
        interests: List[str] = Field(default_factory=list, description="Academic & personal interests")
        quiz_scores: List[Dict[str, Any]] = Field(default_factory=list, description="Historical quiz performance")
        readiness_score: float = Field(default=50.0, description="Overall exam readiness score (0-100)")
        last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

    class ChatMessage(BaseModel):
        """Structured record of a single chat turn."""
        role: str = Field(..., description="'user', 'assistant', or 'system'")
        content: str = Field(..., description="Message text content")
        timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        citations: List[Dict[str, Any]] = Field(default_factory=list, description="RAG source citations")
        tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Executed tool metadata")

    class RAGChunk(BaseModel):
        """Retrieved document chunk representation."""
        doc_id: str = Field(..., description="Unique document hash or ID")
        file_name: str = Field(..., description="Source file name")
        content: str = Field(..., description="Chunk text content")
        page_number: Optional[int] = Field(default=None, description="Page number if applicable")
        relevance_score: float = Field(default=0.0, description="Similarity score")

    class ToolCallResult(BaseModel):
        """Result payload from tool execution."""
        tool_name: str = Field(..., description="Executed tool name")
        status: str = Field(default="success", description="'success' or 'error'")
        result_data: Dict[str, Any] = Field(default_factory=dict, description="Structured tool outputs")
        rendered_output: str = Field(default="", description="Markdown or text display string")
        error_message: Optional[str] = Field(default=None, description="Error details if failed")
else:
    class LearnerProfile:
        pass
    class ChatMessage:
        pass
    class RAGChunk:
        pass
    class ToolCallResult:
        pass


class GraphState(TypedDict):
    """
    Core state object passed across LangGraph nodes.
    Maintains user query context, chat history, extracted memory, RAG documents,
    tool execution state, and final output response.
    """
    user_id: str
    user_query: str
    intent: str
    intent_confidence: float
    chat_history: List[Dict[str, str]]
    learner_profile: Dict[str, Any]
    retrieved_docs: List[Dict[str, Any]]
    active_tool: Optional[str]
    tool_args: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    generated_response: str
    citations: List[Dict[str, Any]]
    memory_updates: Dict[str, Any]
    error: Optional[str]
