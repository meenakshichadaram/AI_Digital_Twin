"""
AI Digital Twin of Knowledge - Dual Memory System

Implements short-term conversation context tracking, long-term persistent profile
management, entity fact storage, and an automated rule-based & LLM memory update pipeline.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from database import db, DatabaseManager
from config import settings


class MemoryManager:
    """
    Unified manager for Short-Term (Chat History) and Long-Term (Learner Profile & Facts) memory.
    """

    def __init__(self, db_instance: Optional[DatabaseManager] = None):
        """Initialize memory manager with database instance."""
        self.db = db_instance or db

    # ==================== SHORT-TERM MEMORY ====================

    def get_short_term_memory(self, user_id: str = "default_user", limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent chat history turns for session context."""
        return self.db.get_chat_history(user_id=user_id, limit=limit)

    def add_chat_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """Add a single message turn to short-term memory."""
        return self.db.save_chat_message(
            user_id=user_id,
            role=role,
            content=content,
            citations=citations,
            tool_calls=tool_calls
        )

    def clear_short_term_memory(self, user_id: str = "default_user") -> None:
        """Clear recent chat history."""
        self.db.clear_chat_history(user_id=user_id)

    # ==================== LONG-TERM PROFILE & FACTS ====================

    def get_learner_profile(self, user_id: str = "default_user") -> Dict[str, Any]:
        """Fetch persistent learner profile."""
        return self.db.get_learner_profile(user_id=user_id)

    def update_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update learner profile fields in database and return updated profile."""
        self.db.update_learner_profile(user_id=user_id, updates=updates)
        return self.get_learner_profile(user_id=user_id)

    def save_long_term_fact(
        self,
        user_id: str,
        category: str,
        fact_key: str,
        fact_value: str,
        confidence: float = 1.0
    ) -> None:
        """Save a long-term extracted fact or observation."""
        self.db.save_fact(
            user_id=user_id,
            category=category,
            fact_key=fact_key,
            fact_value=fact_value,
            confidence=confidence
        )

    def get_long_term_facts(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Retrieve all extracted long-term facts for a learner."""
        return self.db.get_facts(user_id=user_id)

    # ==================== MEMORY CONTEXT GENERATOR ====================

    def get_formatted_memory_context(self, user_id: str = "default_user") -> str:
        """
        Build a concise, structured memory summary string suitable for injection
        into LLM system prompts.
        """
        profile = self.get_learner_profile(user_id)
        facts = self.get_long_term_facts(user_id)

        lines = ["=== LEARNER DIGITAL TWIN PROFILE ==="]
        lines.append(f"Name: {profile.get('name', 'Learner')}")
        if profile.get("age"):
            lines.append(f"Age: {profile.get('age')}")
        if profile.get("college"):
            lines.append(f"College/Inst: {profile.get('college')}")
        if profile.get("university"):
            lines.append(f"University: {profile.get('university')}")
        if profile.get("branch"):
            lines.append(f"Branch/Major: {profile.get('branch')}")

        # Learning preferences & metrics
        lines.append(f"Preferred Learning Style: {profile.get('preferred_learning_style', 'Visual & Practical')}")
        lines.append(f"Daily Target Study Hours: {profile.get('daily_study_hours', 3.0)} hrs")
        lines.append(f"Current Exam Readiness Score: {profile.get('readiness_score', 50.0):.1f}%")

        # Subjects & Goals
        subjects = profile.get("subjects", [])
        if subjects:
            lines.append(f"Active Subjects: {', '.join(subjects)}")

        goals = profile.get("learning_goals", [])
        if goals:
            lines.append(f"Learning Goals: {', '.join(goals)}")

        # Weak and Strong Concepts
        weak = profile.get("weak_concepts", [])
        if weak:
            lines.append(f"Weak Concepts (Needs Revision): {', '.join(weak)}")

        strong = profile.get("strong_concepts", [])
        if strong:
            lines.append(f"Mastered Concepts: {', '.join(strong)}")

        # Exam Schedules
        exam_dates = profile.get("exam_dates", {})
        if exam_dates:
            schedule_str = ", ".join([f"{sub}: {dt}" for sub, dt in exam_dates.items()])
            lines.append(f"Exam Deadlines: {schedule_str}")

        # Extracted Long term facts
        if facts:
            lines.append("--- Extracted Personal Observations & Facts ---")
            for f in facts[:10]:
                lines.append(f"- [{f.get('category', 'FACT')}] {f.get('fact_key')}: {f.get('fact_value')}")

        return "\n".join(lines)

    # ==================== AUTOMATED MEMORY UPDATE PIPELINE ====================

    def auto_extract_and_update_memory(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str = ""
    ) -> Dict[str, Any]:
        """
        Rule-based and pattern extraction engine that identifies personal statements
        in user messages (name, weak concepts, exam dates, college, study hours)
        and automatically updates the persistent SQLite database.
        Returns dictionary of updated attributes.
        """
        profile = self.get_learner_profile(user_id)
        updates: Dict[str, Any] = {}
        extracted_facts: List[Tuple[str, str, str]] = []  # (category, key, value)

        text = user_message.strip()

        # 1. Name Extraction ("My name is Namith", "I am Namith", "Call me Namith")
        name_match = re.search(r"(?:my name is|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)", text, re.IGNORECASE)
        if name_match:
            raw_name = name_match.group(1).strip()
            # Stop before conjunctions or prepositions if present
            clean_name = re.split(r"\s+(?:and|or|i|studying|at|in|from|with)\b", raw_name, flags=re.IGNORECASE)[0]
            if clean_name.lower() not in ["weak", "good", "confused", "studying", "preparing", "happy", "learning", "a", "the"]:
                updates["name"] = clean_name.capitalize()
                extracted_facts.append(("PROFILE", "Name", clean_name.capitalize()))

        # 2. Weak Concept Extraction ("I am weak in trees and graphs", "im weak in movies", "trouble with calculus")
        weak_match = re.search(r"(?:weak in|weak at|struggle with|don't understand|find.*difficult|trouble with|need help with)\s+([a-zA-Z0-9\s_\-\+\&]{2,60})", text, re.IGNORECASE)
        if weak_match:
            raw_concept = weak_match.group(1).strip().strip(".,!?")
            # Clean trailing intent keywords
            concept = re.split(r"\s+(?:i\s+want|because|since|explain|master)\b", raw_concept, flags=re.IGNORECASE)[0].strip()
            if "tree" in concept.lower() or "graph" in concept.lower():
                concept = "Trees and Graphs in DSA"
            else:
                concept = concept.title()
            
            existing_weak = profile.get("weak_concepts", [])
            existing_lower = [w.lower() for w in existing_weak]
            if concept and concept.lower() not in existing_lower:
                new_weak = list(existing_weak) + [concept]
                updates["weak_concepts"] = new_weak
                extracted_facts.append(("WEAKNESS", "Weak Topic", concept))

        # 3. Strong Concept Extraction ("I master trees and graphs", "I am good at movies", "I understand data structures")
        strong_match = re.search(r"(?:good at|mastered|understand|expert in|excel in|master)\s+([a-zA-Z0-9\s_\-\+\&]{2,60})", text, re.IGNORECASE)
        if strong_match and "want to master" not in text.lower():
            raw_concept = strong_match.group(1).strip().strip(".,!?")
            concept = re.split(r"\s+(?:because|since|also|explain)\b", raw_concept, flags=re.IGNORECASE)[0].strip()
            if "tree" in concept.lower() or "graph" in concept.lower():
                concept = "Trees and Graphs in DSA"
            else:
                concept = concept.title()

            existing_strong = profile.get("strong_concepts", [])
            existing_s_lower = [s.lower() for s in existing_strong]
            if concept and concept.lower() not in existing_s_lower:
                new_strong = list(existing_strong) + [concept]
                updates["strong_concepts"] = new_strong
                # Remove from weak if now mastered
                existing_weak = profile.get("weak_concepts", [])
                updates["weak_concepts"] = [w for w in existing_weak if w.lower() != concept.lower()]
                extracted_facts.append(("STRENGTH", "Mastered Topic", concept))

        # 4. Exam Date / Deadline Extraction ("My exam is next week", "Exam on 15th August", "Finals in 3 days")
        exam_match = re.search(r"(?:exam|test|finals|midterm)(?: is)?\s+(?:on|in|next)\s+([a-zA-Z0-9\s]{3,30})", text, re.IGNORECASE)
        if exam_match:
            raw_date = exam_match.group(1).strip().strip(".,!?")
            date_str = re.split(r"\s+(?:and|or|so|i|which)\b", raw_date, flags=re.IGNORECASE)[0].strip()
            existing_exams = profile.get("exam_dates", {})
            existing_exams["Upcoming Exam"] = date_str
            updates["exam_dates"] = existing_exams
            extracted_facts.append(("DEADLINE", "Upcoming Exam", date_str))

        # 5. College / Branch Extraction ("I study computer science at MIT", "I am in VTU university")
        college_match = re.search(r"(?:study|student|enrolled)\s+(?:at|in)\s+([A-Za-z0-9\s]{3,40}\s+(?:College|University|Institute|IIT|NIT|VTU|BITS|MIT))", text, re.IGNORECASE)
        if college_match:
            college_name = college_match.group(1).strip()
            updates["college"] = college_name
            extracted_facts.append(("PROFILE", "College", college_name))

        branch_match = re.search(r"(?:branch|major|degree|studying)\s+(?:is|in)\s+([A-Za-z0-9\s]{3,30}(?:Engineering|Computer Science|AI|DS|Physics|Math|ECE|EEE|Mechanical))", text, re.IGNORECASE)
        if branch_match:
            branch_name = branch_match.group(1).strip()
            updates["branch"] = branch_name
            extracted_facts.append(("PROFILE", "Branch", branch_name))

        # 6. Daily Study Hours ("I study 4 hours a day", "I can spend 2.5 hours daily")
        hours_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours|hrs)\s*(?:a day|daily|per day)", text, re.IGNORECASE)
        if hours_match:
            try:
                hrs = float(hours_match.group(1))
                updates["daily_study_hours"] = hrs
                extracted_facts.append(("HABIT", "Daily Study Hours", f"{hrs} hrs"))
            except ValueError:
                pass

        # Apply profile updates to database if any discovered
        if updates:
            self.update_profile(user_id=user_id, updates=updates)

        # Save extracted long-term facts
        for cat, key, val in extracted_facts:
            self.save_long_term_fact(user_id=user_id, category=cat, fact_key=key, fact_value=val)

        return updates


# Global Memory Manager Instance
memory_manager = MemoryManager()
