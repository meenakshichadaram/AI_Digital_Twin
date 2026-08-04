"""
AI Digital Twin of Knowledge - Database Persistence Engine

Manages SQLite database schema creation, CRUD operations for learner profiles,
short-term chat history, long-term extracted facts, quiz performance,
document tracking, and study schedules.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from config import settings


class DatabaseManager:
    """SQLite database manager for persistent learner profile & system memory."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection path and create tables if missing."""
        self.db_path = db_path or settings.DATABASE_PATH
        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Establish and return a SQLite connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create database tables if they do not already exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Learner Profile Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS learner_profile (
                user_id TEXT PRIMARY KEY,
                name TEXT DEFAULT 'Learner',
                age INTEGER,
                college TEXT,
                university TEXT,
                branch TEXT,
                subjects_json TEXT DEFAULT '[]',
                learning_goals_json TEXT DEFAULT '[]',
                exam_dates_json TEXT DEFAULT '{}',
                weak_concepts_json TEXT DEFAULT '[]',
                strong_concepts_json TEXT DEFAULT '[]',
                preferred_learning_style TEXT DEFAULT 'Visual & Practical',
                daily_study_hours REAL DEFAULT 3.0,
                interests_json TEXT DEFAULT '[]',
                readiness_score REAL DEFAULT 50.0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 2. Short Term Memory (Chat Messages History)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS short_term_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                citations_json TEXT DEFAULT '[]',
                tool_calls_json TEXT DEFAULT '[]'
            )
            """)

            # 3. Long Term Memory (Extracted Facts & Attributes)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, category, fact_key) ON CONFLICT REPLACE
            )
            """)

            # 4. Quiz Performance History
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                score INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                percentage REAL NOT NULL,
                details_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 5. Uploaded Document Tracker
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                doc_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                total_chunks INTEGER NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 6. Daily Goals Tracker
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                goal_text TEXT NOT NULL,
                is_completed INTEGER DEFAULT 0,
                target_date DATE DEFAULT (DATE('now')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # 7. Revision Schedules
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS revision_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            conn.commit()

    # ==================== PROFILE METHODS ====================

    def get_learner_profile(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Fetch complete learner profile, creating a default one if not exists."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learner_profile WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()

            if not row:
                # Seed default profile
                cursor.execute("""
                INSERT INTO learner_profile (user_id, name) VALUES (?, ?)
                """, (user_id, "Learner"))
                conn.commit()
                return self.get_learner_profile(user_id)

            profile = dict(row)
            # Parse JSON fields
            profile["subjects"] = json.loads(profile.get("subjects_json") or "[]")
            profile["learning_goals"] = json.loads(profile.get("learning_goals_json") or "[]")
            profile["exam_dates"] = json.loads(profile.get("exam_dates_json") or "{}")
            profile["weak_concepts"] = json.loads(profile.get("weak_concepts_json") or "[]")
            profile["strong_concepts"] = json.loads(profile.get("strong_concepts_json") or "[]")
            profile["interests"] = json.loads(profile.get("interests_json") or "[]")
            return profile

    def update_learner_profile(self, user_id: str, updates: Dict[str, Any]) -> None:
        """Update specific fields in learner profile."""
        existing = self.get_learner_profile(user_id)

        # Merge fields
        name = updates.get("name", existing.get("name"))
        age = updates.get("age", existing.get("age"))
        college = updates.get("college", existing.get("college"))
        university = updates.get("university", existing.get("university"))
        branch = updates.get("branch", existing.get("branch"))
        pref_style = updates.get("preferred_learning_style", existing.get("preferred_learning_style"))
        daily_hours = updates.get("daily_study_hours", existing.get("daily_study_hours"))
        readiness = updates.get("readiness_score", existing.get("readiness_score"))

        # Merge JSON list fields (ensuring uniqueness where appropriate)
        subjects = updates.get("subjects", existing.get("subjects", []))
        goals = updates.get("learning_goals", existing.get("learning_goals", []))
        exam_dates = updates.get("exam_dates", existing.get("exam_dates", {}))
        weak = updates.get("weak_concepts", existing.get("weak_concepts", []))
        strong = updates.get("strong_concepts", existing.get("strong_concepts", []))
        interests = updates.get("interests", existing.get("interests", []))

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE learner_profile SET
                name = ?, age = ?, college = ?, university = ?, branch = ?,
                subjects_json = ?, learning_goals_json = ?, exam_dates_json = ?,
                weak_concepts_json = ?, strong_concepts_json = ?,
                preferred_learning_style = ?, daily_study_hours = ?,
                interests_json = ?, readiness_score = ?,
                last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """, (
                name, age, college, university, branch,
                json.dumps(subjects), json.dumps(goals), json.dumps(exam_dates),
                json.dumps(weak), json.dumps(strong),
                pref_style, daily_hours, json.dumps(interests), readiness,
                user_id
            ))
            conn.commit()

    # ==================== SHORT-TERM MEMORY METHODS ====================

    def save_chat_message(
        self,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Persist a single chat message."""
        citations_json = json.dumps(citations or [])
        tool_calls_json = json.dumps(tool_calls or [])
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO short_term_memory (user_id, role, content, citations_json, tool_calls_json)
            VALUES (?, ?, ?, ?, ?)
            """, (user_id, role, content, citations_json, tool_calls_json))
            conn.commit()
            return cursor.lastrowid

    def get_chat_history(self, user_id: str = "default_user", limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent chat history turns."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT role, content, timestamp, citations_json, tool_calls_json
            FROM short_term_memory
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """, (user_id, limit))
            rows = cursor.fetchall()
            
            history = []
            for r in reversed(rows):
                history.append({
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "citations": json.loads(r["citations_json"] or "[]"),
                    "tool_calls": json.loads(r["tool_calls_json"] or "[]")
                })
            return history

    def clear_chat_history(self, user_id: str = "default_user") -> None:
        """Clear short-term chat history for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM short_term_memory WHERE user_id = ?", (user_id,))
            conn.commit()

    # ==================== LONG-TERM FACTS METHODS ====================

    def save_fact(self, user_id: str, category: str, fact_key: str, fact_value: str, confidence: float = 1.0) -> None:
        """Save or update an extracted long-term memory fact."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO long_term_facts (user_id, category, fact_key, fact_value, confidence)
            VALUES (?, ?, ?, ?, ?)
            """, (user_id, category, fact_key, fact_value, confidence))
            conn.commit()

    def get_facts(self, user_id: str = "default_user", category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve stored long-term facts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("""
                SELECT category, fact_key, fact_value, confidence, created_at
                FROM long_term_facts WHERE user_id = ? AND category = ?
                ORDER BY created_at DESC
                """, (user_id, category))
            else:
                cursor.execute("""
                SELECT category, fact_key, fact_value, confidence, created_at
                FROM long_term_facts WHERE user_id = ?
                ORDER BY created_at DESC
                """, (user_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # ==================== QUIZ & ANALYTICS METHODS ====================

    def record_quiz_result(self, user_id: str, topic: str, score: int, total: int, details: Dict[str, Any]) -> int:
        """Record quiz execution results."""
        pct = (score / total * 100.0) if total > 0 else 0.0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO quiz_results (user_id, topic, score, total_questions, percentage, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, topic, score, total, pct, json.dumps(details)))
            conn.commit()
            return cursor.lastrowid

    def get_quiz_history(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Fetch all quiz attempt histories."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT topic, score, total_questions, percentage, details_json, created_at
            FROM quiz_results WHERE user_id = ? ORDER BY id DESC
            """, (user_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["details"] = json.loads(item.pop("details_json") or "{}")
                results.append(item)
            return results

    # ==================== DOCUMENT TRACKING ====================

    def save_uploaded_document(self, doc_id: str, user_id: str, file_name: str, file_path: str, file_type: str, total_chunks: int) -> None:
        """Record uploaded document metadata."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO uploaded_documents (doc_id, user_id, file_name, file_path, file_type, total_chunks)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET total_chunks = excluded.total_chunks
            """, (doc_id, user_id, file_name, file_path, file_type, total_chunks))
            conn.commit()

    def get_uploaded_documents(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieve list of user uploaded documents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT doc_id, file_name, file_path, file_type, total_chunks, uploaded_at
            FROM uploaded_documents WHERE user_id = ? ORDER BY uploaded_at DESC
            """, (user_id,))
            return [dict(r) for r in cursor.fetchall()]


# Global database manager instance
db = DatabaseManager()
