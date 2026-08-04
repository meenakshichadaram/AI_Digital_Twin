"""
AI Digital Twin of Knowledge - Configuration Manager

Provides centralized management for environment variables, path settings,
LLM configurations, vector store directories, and UI design constants.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass


class Config:
    """System-wide configuration settings with validation and fallbacks."""

    # System Meta
    APP_NAME: str = "AI Digital Twin of Knowledge"
    APP_VERSION: str = "1.0.0"
    AUTHOR: str = "Digital Twin SaaS Platform"

    # Base Paths
    BASE_DIR: Path = BASE_DIR
    MEMORY_DIR: Path = BASE_DIR / "memory"
    VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
    UPLOAD_DIR: Path = BASE_DIR / "uploaded_docs"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    CSS_DIR: Path = BASE_DIR / "assets" / "css"

    # Database & Storage Settings
    DATABASE_PATH: str = os.getenv(
        "DATABASE_PATH", str(BASE_DIR / "memory" / "digital_twin_memory.db")
    )
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR", str(BASE_DIR / "vectorstore" / "chroma_db")
    )

    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # LLM Model Choices
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GOOGLE_MODEL: str = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    # Embedding Config
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Hyperparameters
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "4"))
    CHAT_HISTORY_LIMIT: int = int(os.getenv("MEM_CHAT_HISTORY_LIMIT", "10"))

    # UI Theme Palette (Glassmorphic SaaS Aesthetics)
    THEME_COLORS: Dict[str, str] = {
        "primary": "#6366F1",        # Indigo Accent
        "primary_dark": "#4F46E5",   # Deep Indigo
        "secondary": "#10B981",      # Emerald Green
        "accent": "#F59E0B",         # Warm Amber
        "background_dark": "#0F172A",# Slate 900
        "card_bg_dark": "rgba(30, 41, 59, 0.7)", # Glass Slate
        "text_dark": "#F8FAFC",      # Slate 50
        "background_light": "#F8FAFC",
        "card_bg_light": "rgba(255, 255, 255, 0.8)",
        "text_light": "#0F172A",
    }

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they do not exist."""
        directories = [
            cls.MEMORY_DIR,
            cls.VECTORSTORE_DIR,
            cls.UPLOAD_DIR,
            cls.ASSETS_DIR,
            cls.CSS_DIR,
            cls.BASE_DIR / "utils",
            cls.BASE_DIR / "agents",
            cls.BASE_DIR / "components",
            cls.BASE_DIR / "pages",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_active_model_name(cls) -> str:
        """Return active model string based on provider setting."""
        if cls.LLM_PROVIDER == "google":
            return cls.GOOGLE_MODEL
        elif cls.LLM_PROVIDER == "groq":
            return cls.GROQ_MODEL
        return cls.OPENAI_MODEL

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate API key presence and paths."""
        status = {
            "openai_configured": bool(cls.OPENAI_API_KEY),
            "google_configured": bool(cls.GOOGLE_API_KEY),
            "groq_configured": bool(cls.GROQ_API_KEY),
            "active_provider": cls.LLM_PROVIDER,
            "active_model": cls.get_active_model_name(),
        }
        return status


# Automatically ensure required directories are created on module import
Config.ensure_directories()

# Global settings instance
settings = Config()
