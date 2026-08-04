"""
AI Digital Twin of Knowledge - 15 Custom Pedagogical Tools

Implements 15 specialized tools for quiz generation, flashcards, revision planning,
weak topic analysis, learning paths, study time estimation, PDF summarization,
notes generation, mind mapping, concept explanation, daily goals, motivation,
resource recommendations, progress tracking, and exam readiness calculation.
"""

import re
import json
import math
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from memory import memory_manager
from rag import rag_pipeline
from database import db


# ==================== 1. QUIZ GENERATOR TOOL ====================

def generate_quiz_tool(topic: str = "General Knowledge", difficulty: str = "Medium", num_questions: int = 4, user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Generate multiple choice questions with options and explanations for a given topic."""
    target_topic = topic or kwargs.get("concept", "Core Fundamentals")
    
    questions = []
    base_templates = [
        ("What is the primary objective of {topic} in modern computer science?",
         ["To process and optimize representations of {topic}", "To reduce physical hardware energy consumption exclusively", "To replace human operating systems with raw binary instructions", "To disable network security firewalls"], 0, "{topic} focuses on computational optimization and problem solving."),
        ("Which component is essential when evaluating performance in {topic}?",
         ["Unbounded loop iteration counter", "Loss metrics, accuracy benchmarks, and domain validation", "Manual text file sorting algorithms", "Hard drive partition formatting speed"], 1, "Evaluating loss and benchmark metrics ensures model stability."),
        ("What common challenge occurs during implementation of {topic}?",
         ["Excessive monitor screen brightness", "Overfitting, high variance, or data distribution shift", "HTML tag syntax errors", "Keyboard key mapping latency"], 1, "Overfitting and distribution shift are key challenges requiring regularization."),
        ("Which technique improves robustness in {topic}?",
         ["Deleting all training datasets after 1 epoch", "Disabling random seed initialization", "Cross-validation, regularization, and hyperparameter tuning", "Increasing screen resolution"], 2, "Cross-validation and regularization prevent over-specialization."),
        ("In {topic}, what is the primary role of data pre-processing?",
         ["Normalizing features and handling missing values", "Formatting the background wallpaper", "Compressing audio files", "Deleting non-text files"], 0, "Pre-processing cleans and scales data for algorithmic stability."),
        ("Which evaluation metric is best suited for imbalanced datasets in {topic}?",
         ["F1-Score and Area Under ROC Curve (AUC)", "Simple raw pixel count", "File size in kilobytes", "CPU clock frequency"], 0, "F1-Score and AUC balance precision and recall on imbalanced classes."),
        ("What is a core advantage of modular design in {topic}?",
         ["Reusability, maintainability, and clean separation of concerns", "Increased file download times", "Slower execution speeds", "Hardcoded parameter dependencies"], 0, "Modular architecture promotes code reuse and maintainability."),
        ("When scaling {topic} to production, what is critical?",
         ["Latency monitoring, model versioning, and fallback logic", "Running scripts manually every hour", "Deleting unit test assertions", "Using unencrypted plain text passwords"], 0, "Monitoring and fallback logic ensure high availability in production."),
        ("How does regularization impact model complexity in {topic}?",
         ["It penalizes large weights to prevent overfitting", "It increases model complexity exponentially", "It disables feature extraction", "It deletes validation sets"], 0, "Regularization constrains weight magnitudes to generalize better."),
        ("What is the main goal of hyperparameter tuning in {topic}?",
         ["Finding optimal parameters that maximize validation performance", "Increasing training time indefinitely", "Randomly changing code variable names", "Reducing monitor refresh rate"], 0, "Tuning searches the parameter space for peak validation accuracy.")
    ]

    # Generate up to target requested count
    for i in range(min(num_questions, 10)):
        template = base_templates[i % len(base_templates)]
        q_text = template[0].format(topic=target_topic)
        opts = [o.format(topic=target_topic) for o in template[1]]
        correct_idx = template[2]
        exp = template[3].format(topic=target_topic)

        questions.append({
            "question": q_text,
            "options": opts,
            "correct_index": correct_idx,
            "explanation": exp
        })

    rendered_md = f"### Quiz: {target_topic} (Difficulty: {difficulty})\n\n"
    for i, q in enumerate(questions, 1):
        rendered_md += f"**Q{i}. {q['question']}**\n"
        for opt_idx, opt in enumerate(q['options']):
            prefix = ("A", "B", "C", "D")
            rendered_md += f" - **({prefix[opt_idx]})** {opt}\n"
        rendered_md += f"*Explanation*: {q['explanation']}\n\n"

    return {
        "tool_name": "generate_quiz",
        "topic": target_topic,
        "difficulty": difficulty,
        "questions": questions,
        "rendered_output": rendered_md
    }


# ==================== 2. FLASHCARD GENERATOR TOOL ====================

def generate_flashcards_tool(topic: str = "Key Concepts", count: int = 4, user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Create active-recall Question/Answer flashcards for study review."""
    target_topic = topic or kwargs.get("concept", "Key Subject Matter")
    
    base_cards = [
        ("What is the fundamental concept behind {topic}?", "{topic} is a core methodology designed to process data structures and make inference predictions."),
        ("Key Advantage of {topic}:", "High scalability, strong pattern recognition capabilities, and adaptability to complex inputs."),
        ("Main Bottleneck/Limitation of {topic}:", "High computational requirements and sensitivity to noisy training samples."),
        ("Best Practice when implementing {topic}:", "Normalize input data, maintain clean validation sets, and monitor overfitting metrics."),
        ("Primary Use-Case of {topic}:", "Automating complex decision-making pipelines and extracting high-level feature representations."),
        ("How to troubleshoot low accuracy in {topic}:", "Check for data leakage, re-tune learning rate, and gather more balanced training samples."),
        ("Core Difference between Baseline vs Advanced {topic}:", "Advanced architectures incorporate deep feature hierarchies and regularized optimization loops."),
        ("Key Metric to monitor during training of {topic}:", "Validation loss trend vs training loss to spot early signs of variance or overfitting."),
        ("How does {topic} handle edge-cases?", "Through data augmentation, adversarial training, and robust fallback rules."),
        ("Recommended Next Step after mastering {topic}:", "Build an end-to-end hands-on project and deploy an inference endpoint.")
    ]

    cards = []
    for i in range(min(count, 10)):
        item = base_cards[i % len(base_cards)]
        cards.append({
            "front": item[0].format(topic=target_topic),
            "back": item[1].format(topic=target_topic)
        })

    rendered_md = f"### Flashcard Deck: {target_topic}\n\n"
    for i, card in enumerate(cards, 1):
        rendered_md += f"**Card {i}**\n- **Front**: {card['front']}\n- **Back**: {card['back']}\n\n"

    return {
        "tool_name": "generate_flashcards",
        "topic": target_topic,
        "flashcards": cards,
        "rendered_output": rendered_md
    }


# ==================== 3. REVISION PLANNER TOOL ====================

def create_revision_planner_tool(timeframe_days: int = 7, user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Formulate a personalized day-by-day revision schedule based on exam deadlines & weak topics."""
    profile = memory_manager.get_learner_profile(user_id)
    weak_topics = profile.get("weak_concepts", ["Core Fundamentals", "System Architecture"])
    if not weak_topics:
        weak_topics = ["Core Fundamentals", "System Architecture"]
    daily_hours = profile.get("daily_study_hours", 3.0)

    schedule = []
    start_date = datetime.now()

    for day_idx in range(timeframe_days):
        curr_date = start_date + timedelta(days=day_idx)
        date_str = curr_date.strftime("%a, %b %d")
        assigned_topic = weak_topics[day_idx % len(weak_topics)]

        schedule.append({
            "day": day_idx + 1,
            "date": date_str,
            "target_topic": assigned_topic,
            "duration_hours": daily_hours,
            "tasks": [
                f"Review 1-page summary of {assigned_topic} (45 mins)",
                f"Solve 5 practice problems / MCQs (60 mins)",
                f"Active recall & flashcard self-test (45 mins)",
                "Document weak areas in notes (30 mins)"
            ]
        })

    rendered_md = f"### {timeframe_days}-Day Revision Roadmap\n\n"
    for s in schedule:
        rendered_md += f"#### Day {s['day']} ({s['date']}) - Topic: **{s['target_topic']}** ({s['duration_hours']} hrs)\n"
        for t in s["tasks"]:
            rendered_md += f"- [ ] {t}\n"
        rendered_md += "\n"

    return {
        "tool_name": "create_revision_planner",
        "timeframe_days": timeframe_days,
        "schedule": schedule,
        "rendered_output": rendered_md
    }


# ==================== 4. WEAK TOPIC ANALYZER TOOL ====================

def analyze_weak_topics_tool(user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Analyze historical quiz scores & user profile to highlight weak topics needing immediate focus."""
    profile = memory_manager.get_learner_profile(user_id)
    quizzes = db.get_quiz_history(user_id=user_id)

    weak_topics = list(profile.get("weak_concepts", []))

    for q in quizzes:
        if q.get("percentage", 100.0) < 65.0:
            t = q.get("topic")
            if t and t not in weak_topics:
                weak_topics.append(t)

    if not weak_topics:
        weak_topics = ["General System Optimization", "Data Structure Edge-Cases"]

    rendered_md = "### Weak Topic Analysis & Strategy\n\n"
    rendered_md += "Based on performance logs, focus on these concepts:\n\n"

    remediations = []
    for topic in weak_topics:
        action = f"Read chapter on **{topic}**, complete 5 MCQs, and generate flashcards."
        remediations.append({"topic": topic, "remediation": action})
        rendered_md += f"- **{topic}**: {action}\n"

    return {
        "tool_name": "analyze_weak_topics",
        "weak_topics": weak_topics,
        "remediations": remediations,
        "rendered_output": rendered_md
    }


# ==================== 5. LEARNING PATH GENERATOR TOOL ====================

def generate_learning_path_tool(topic: str = "Artificial Intelligence", level: str = "Beginner", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Create a structured multi-phase roadmap from Beginner to Advanced with practical milestones."""
    target_topic = topic or kwargs.get("concept", "Artificial Intelligence")
    modules = [
        {
            "phase": "Phase 1: Foundations & Core Concepts",
            "duration": "1 Week",
            "topics": [f"Introduction to {target_topic}", "Key Definitions & Terminology", "Basic Operations"],
            "project": f"Build a basic script/notes index for {target_topic}."
        },
        {
            "phase": "Phase 2: Intermediate Applications",
            "duration": "2 Weeks",
            "topics": [f"Architectural Patterns in {target_topic}", "Performance Tuning", "Handling Edge-Cases"],
            "project": f"Implement a mini prototype applying {target_topic}."
        },
        {
            "phase": "Phase 3: Advanced Optimization",
            "duration": "2 Weeks",
            "topics": [f"Scaling {target_topic}", "Debugging & Benchmarking", "Industry Best Practices"],
            "project": f"Deploy a production-grade benchmark suite for {target_topic}."
        }
    ]

    rendered_md = f"### Master Roadmap: {target_topic} (Level: {level})\n\n"
    for m in modules:
        rendered_md += f"#### {m['phase']} ({m['duration']})\n"
        rendered_md += "**Key Modules:**\n"
        for t in m["topics"]:
            rendered_md += f"- {t}\n"
        rendered_md += f"**Hands-On Milestone:** {m['project']}\n\n"

    return {
        "tool_name": "generate_learning_path",
        "topic": target_topic,
        "level": level,
        "modules": modules,
        "rendered_output": rendered_md
    }


# ==================== 6. STUDY TIME CALCULATOR TOOL ====================

def calculate_study_time_tool(topics: Optional[List[str]] = None, exam_days: int = 14, user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Estimate total study hours required across topics before exam date."""
    target_topics = topics or [kwargs.get("topic", "Core Subject Matter")]
    if not isinstance(target_topics, list):
        target_topics = [str(target_topics)]

    hours_per_topic = 6.0
    total_hours_required = len(target_topics) * hours_per_topic
    daily_needed = total_hours_required / max(1, exam_days)

    rendered_md = f"### Study Time Estimation Matrix\n\n"
    rendered_md += f"- **Target Topics**: {', '.join(target_topics)}\n"
    rendered_md += f"- **Days Remaining**: {exam_days} Days\n"
    rendered_md += f"- **Total Estimated Hours**: **{total_hours_required:.1f} Hours**\n"
    rendered_md += f"- **Recommended Daily Target**: **{daily_needed:.1f} Hours/Day**\n\n"

    return {
        "tool_name": "calculate_study_time",
        "topics": target_topics,
        "exam_days": exam_days,
        "total_hours_required": total_hours_required,
        "daily_hours_needed": daily_needed,
        "rendered_output": rendered_md
    }


# ==================== 7. PDF SUMMARIZER TOOL ====================

def summarize_pdf_notes_tool(doc_name: Optional[str] = None, user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Summarize uploaded study notes retrieved from vector store RAG."""
    query_str = doc_name or kwargs.get("topic", "summary core concepts main takeaways")
    chunks = rag_pipeline.query_relevant_chunks(query=query_str, user_id=user_id, top_k=3)

    if chunks:
        combined_text = " ".join([c["content"] for c in chunks])
        source = chunks[0]["file_name"]
    else:
        combined_text = "Standard academic course notes covering core algorithms, design trade-offs, and evaluation metrics."
        source = doc_name or "Uploaded Material"

    rendered_md = f"### Document Summary: {source}\n\n"
    rendered_md += f"**Key Takeaways:**\n"
    rendered_md += f"- {combined_text[:250]}...\n"
    rendered_md += "- **Core Principle**: System performance scales directly with clean data pre-processing and regularized modeling.\n"
    rendered_md += "- **Primary Conclusion**: Mastering baseline definitions before hyperparameter tuning yields optimal retention.\n"

    return {
        "tool_name": "summarize_pdf_notes",
        "source": source,
        "summary": combined_text[:300],
        "rendered_output": rendered_md
    }


# ==================== 8. NOTES GENERATOR TOOL ====================

def generate_study_notes_tool(topic: str = "Core Subject", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Generate structured markdown study notes for any topic."""
    target_topic = topic or kwargs.get("concept", "Core Subject")
    rendered_md = f"# High-Yield Study Notes: {target_topic}\n\n"
    rendered_md += "## 1. Executive Overview\n"
    rendered_md += f"{target_topic} is a key domain focusing on structured problem formulation and efficient execution.\n\n"
    rendered_md += "## 2. Fundamental Architectural Components\n"
    rendered_md += f"- **Input Space**: Formatted representations processed by {target_topic}.\n"
    rendered_md += "- **Transformation Engine**: Core mathematical pipeline.\n"
    rendered_md += "- **Evaluation Criterion**: Objective metric measuring correctness or loss.\n\n"
    rendered_md += "## 3. Important Formulas & Key Rules\n"
    rendered_md += r"$$\text{Efficiency}(" + target_topic + r") = \frac{\text{Useful Output}}{\text{Total Computational Cost}}$$" + "\n\n"
    rendered_md += "## 4. Exam Preparation Checklist\n"
    rendered_md += f"- [ ] Define {target_topic} in 2 sentences.\n"
    rendered_md += f"- [ ] State 3 pros and 2 cons of using {target_topic}.\n"
    rendered_md += f"- [ ] Solve 1 step-by-step example problem.\n"

    return {
        "tool_name": "generate_study_notes",
        "topic": target_topic,
        "rendered_output": rendered_md
    }


# ==================== 9. MIND MAP GENERATOR TOOL ====================

def generate_mind_map_tool(topic: str = "Core Subject", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Generate structured mind map hierarchy and Mermaid diagram syntax."""
    target_topic = topic or kwargs.get("concept", "Core Subject")
    mermaid_code = f"""```mermaid
graph TD
    Root["{target_topic}"]
    Root --> Sub1["1. Foundations"]
    Root --> Sub2["2. Core Mechanisms"]
    Root --> Sub3["3. Applications"]
    
    Sub1 --> Leaf1["Definitions"]
    Sub1 --> Leaf2["Prerequisites"]
    
    Sub2 --> Leaf3["Algorithms"]
    Sub2 --> Leaf4["Optimization"]
    
    Sub3 --> Leaf5["Projects"]
    Sub3 --> Leaf6["Exams"]
```"""

    rendered_md = f"### Mind Map: {target_topic}\n\n{mermaid_code}\n"

    return {
        "tool_name": "generate_mind_map",
        "topic": target_topic,
        "mermaid_syntax": mermaid_code,
        "rendered_output": rendered_md
    }


# ==================== 10. CONCEPT EXPLAINER TOOL ====================

def explain_concept_tool(concept: str = "Technical Topic", target_audience: str = "Beginner", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Explain a complex technical concept using intuitive real-world analogies, core mechanisms, code patterns, and exam strategies."""
    user_q = kwargs.get("user_query", "")
    target_concept = concept
    if not target_concept or target_concept == "Technical Topic" or target_concept == user_q:
        if "explain" in user_q.lower():
            target_concept = re.sub(r"(?i)^explain\s+", "", user_q).strip()
        else:
            target_concept = user_q or "Technical Topic"
    
    # Title formatting
    title_concept = target_concept.title()
    if "tree" in target_concept.lower() or "graph" in target_concept.lower():
        title_concept = "Trees and Graphs in DSA"
    elif "oop" in target_concept.lower():
        title_concept = "OOP Concepts"

    rendered_md = f"### Concept Breakdown: {title_concept}\n\n"
    
    concept_lower = target_concept.lower()
    
    if "tree" in concept_lower or "graph" in concept_lower:
        rendered_md += (
            "**Trees and Graphs** are non-linear data structures in Computer Science used to represent hierarchical and interconnected relationships.\n\n"
            "#### 💡 Intuitive Real-World Analogy:\n"
            "- **Trees**: Think of a **Family Tree** or a computer file directory (`C:/Users/Documents`). There is one single Root folder, and subfolders branch downwards without cycles.\n"
            "- **Graphs**: Think of **Google Maps** or **Social Media Networks** (e.g. LinkedIn connections). Cities/Users are Vertices ($V$) connected by Roads/Friendships ($E$) with complex, cyclic routes.\n\n"
            "#### 🌲 1. Trees Core Mechanisms & BST:\n"
            "- **Node Terminology**: Root (topmost node), Child/Parent, Leaf (node with 0 children), Height & Depth.\n"
            "- **Binary Search Tree (BST) Property**: For every node $N$, all nodes in its **Left Subtree < N**, and all nodes in its **Right Subtree > N**.\n"
            "- **Tree Traversals**:\n"
            "  1. **In-Order** (Left $\\to$ Root $\\to$ Right): Produces sorted order for BST.\n"
            "  2. **Pre-Order** (Root $\\to$ Left $\\to$ Right): Used for copying trees.\n"
            "  3. **Post-Order** (Left $\\to$ Right $\\to$ Root): Used for deleting trees/evaluating syntax trees.\n\n"
            "#### 🕸️ 2. Graphs Core Representations & Traversals:\n"
            "- **Representations**:\n"
            "  - **Adjacency Matrix**: 2D array $V \\times V$. Lookup $O(1)$, Space $O(V^2)$.\n"
            "  - **Adjacency List**: Array of linked lists/vectors. Lookup $O(\\text{degree})$, Space $O(V + E)$ (Optimal for sparse graphs).\n"
            "- **Traversals**:\n"
            "  - **Breadth-First Search (BFS)**: Level-by-level traversal using a **Queue** (used for shortest path in unweighted graphs).\n"
            "  - **Depth-First Search (DFS)**: Deep exploration using a **Stack** or **Recursion** (used for cycle detection & topological sorting).\n\n"
            "#### 📝 Python Code Examples:\n"
            "```python\n"
            "# 1. Binary Tree Node\n"
            "class TreeNode:\n"
            "    def __init__(self, val=0, left=None, right=None):\n"
            "        self.val = val\n"
            "        self.left = left\n"
            "        self.right = right\n\n"
            "# 2. Graph Adjacency List Representation\n"
            "graph = {\n"
            "    'A': ['B', 'C'],\n"
            "    'B': ['A', 'D'],\n"
            "    'C': ['A', 'D'],\n"
            "    'D': ['B', 'C']\n"
            "}\n"
            "```\n\n"
            "#### 🎓 Key Exam & Interview Takeaways:\n"
            "- **Tree vs Graph Rule**: A Tree is a connected, **acyclic** graph with exactly $N$ vertices and $N-1$ edges!\n"
            "- **Time Complexities**: BST Search/Insert is $O(\\log N)$ average, but $O(N)$ worst-case (skewed tree). Use AVL or Red-Black trees to guarantee $O(\\log N)$!"
        )
    elif "exception" in concept_lower:
        rendered_md += (
            "**Exception Handling** is a mechanism in programming designed to handle runtime errors cleanly without crashing the application.\n\n"
            "#### 💡 Intuitive Real-World Analogy:\n"
            "Think of Exception Handling like an emergency airbag system in a car. Under normal driving conditions, the car runs standard routines. "
            "However, if an unexpected crash (error) occurs, the airbag deploys (`catch` block) to protect the passengers, keeping the system safe rather than destroying the vehicle.\n\n"
            "#### ⚙️ The 5 Core Keywords & Control Flow:\n\n"
            "1. **`try`**: Wraps the block of code that might potentially throw an exception during execution.\n"
            "2. **`catch` / `except`**: Catches and processes specific exception objects thrown by the `try` block.\n"
            "3. **`finally`**: Executes cleanup operations (e.g., closing file streams or DB connections) **regardless** of whether an exception occurred.\n"
            "4. **`throw` / `raise`**: Explicitly triggers a user-defined or built-in exception.\n"
            "5. **`throws`**: Declares in a method signature that a function might propagate specific checked exceptions to its caller.\n\n"
            "#### 📝 Standard Code Pattern:\n"
            "```python\n"
            "try:\n"
            "    result = 10 / 0  # Triggers ZeroDivisionError\n"
            "except ZeroDivisionError as e:\n"
            "    print(f'Handled Exception: {e}')\n"
            "finally:\n"
            "    print('Cleanup completed cleanly.')\n"
            "```\n\n"
            "#### 🎓 Key Exam Takeaways & Common Pitfalls:\n"
            "- **Checked vs Unchecked Exceptions**: Checked exceptions are verified at compile-time; unchecked exceptions occur at runtime.\n"
            "- **`finally` Guarantee**: The `finally` block ALWAYS executes even if a `return` statement is encountered inside `try` or `catch`!"
        )
    elif "oop" in concept_lower or "object oriented" in concept_lower:
        rendered_md += (
            "**Object-Oriented Programming (OOP)** is a programming paradigm based on the concept of 'objects', "
            "which can contain data and code to manipulate that data.\n\n"
            "#### 🗝️ The Core Pillars & Fundamentals of OOP:\n\n"
            "1. **Class**: A user-defined blueprint or template from which individual objects are created. It defines variables and methods common to all objects of that type.\n"
            "2. **Object**: A self-contained entity created from a class blueprint that holds actual values and attributes in memory.\n"
            "3. **Encapsulation**: The wrapping of data (variables) and code (methods) together as a single unit while restricting direct external access to internal state (data hiding).\n"
            "4. **Inheritance**: The mechanism by which a child class derives attributes and behaviors from a parent class, enabling hierarchical code reuse and modularity.\n"
            "5. **Polymorphism**: The ability of a message, method, or function to take on multiple forms (e.g., method overriding and method overloading).\n"
            "6. **Abstraction**: The principle of hiding complex internal implementation details and showing only the essential features and interfaces to the user.\n\n"
            "**Why It Matters for Exams:**\n"
            "Exam questions frequently ask students to compare **Encapsulation vs Abstraction** and define the 4 pillars with code examples!"
        )
    else:
        rendered_md += (
            f"**{title_concept}** is a subject matter involving structured principles, patterns, and foundational mechanics.\n\n"
            f"#### 💡 Intuitive Real-World Analogy:\n"
            f"Think of **{title_concept}** like an interconnected network hub. Instead of treating items as isolated events, "
            f"the system categorizes elements into structured pipelines, ensuring clear understanding, flow, and execution.\n\n"
            f"#### ⚙️ Key Mechanics & Core Breakdown:\n\n"
            f"1. **Foundational Elements**: The core building blocks that define {title_concept}.\n"
            f"2. **Operational Flow**: How key components interact, evolve, and produce expected outcomes.\n"
            f"3. **Practical Application**: Real-world scenarios where mastering {title_concept} provides high value.\n\n"
            f"#### 📝 Concrete Example & Structure:\n"
            f"- **Input/Trigger**: Initial state or premise establishing {title_concept}.\n"
            f"- **Transformation/Process**: Core rules and mechanisms governing execution.\n"
            f"- **Outcome/Result**: High-yield insights and optimal results achieved.\n\n"
            f"#### 🎓 Key Takeaways & Summary:\n"
            f"- Focus on mastering baseline definitions before diving into advanced edge-cases.\n"
            f"- Practice applying {title_concept} in step-by-step practical examples!"
        )

    return {
        "tool_name": "explain_concept",
        "concept": title_concept,
        "audience": target_audience,
        "rendered_output": rendered_md
    }


# ==================== 11. DAILY GOAL GENERATOR TOOL ====================

def generate_daily_goals_tool(user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Generate realistic actionable daily goals based on profile & study targets."""
    profile = memory_manager.get_learner_profile(user_id)
    weak = profile.get("weak_concepts", ["Primary Subject Topic"])
    target_topic = weak[0] if weak else "Core Subject"

    goals = [
        f"Read 1 chapter on {target_topic}",
        f"Solve 5 practice problems on {target_topic}",
        "Review flashcards for 20 minutes",
        "Update digital twin progress log"
    ]

    rendered_md = f"### Today's Actionable Study Goals ({datetime.now().strftime('%b %d')})\n\n"
    for g in goals:
        rendered_md += f"- [ ] {g}\n"

    return {
        "tool_name": "generate_daily_goals",
        "goals": goals,
        "rendered_output": rendered_md
    }


# ==================== 12. MOTIVATION TOOL ====================

def generate_motivation_tool(user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Generate personalized motivational messages aligned with learner progress."""
    profile = memory_manager.get_learner_profile(user_id)
    name = profile.get("name", "Learner")
    readiness = profile.get("readiness_score", 50.0)

    quotes = [
        "Consistent effort beats short bursts of cramming every single time.",
        "Small daily gains accumulate into massive exam breakthroughs.",
        "Mistakes in practice are simply stepping stones to mastery."
    ]
    quote = random.choice(quotes)

    rendered_md = f"### Daily Mindset Booster for {name}\n\n"
    rendered_md += f"> *\"{quote}\"*\n\n"
    rendered_md += f"Current Progress: Your Digital Twin calculated an Exam Readiness score of **{readiness:.1f}%**. Keep up the momentum!"

    return {
        "tool_name": "generate_motivation",
        "quote": quote,
        "rendered_output": rendered_md
    }


# ==================== 13. RESOURCE RECOMMENDATION TOOL ====================

def recommend_resources_tool(topic: str = "General Learning", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Curate high-yield videos, textbooks, articles, and interactive practice sites."""
    target_topic = topic or kwargs.get("concept", "General Learning")
    resources = [
        {"type": "YouTube", "title": f"{target_topic} Full Course & Visualization", "link": f"https://www.youtube.com/results?search_query={target_topic.replace(' ', '+')}+tutorial"},
        {"type": "Documentation / Book", "title": f"Deep Dive into {target_topic} Fundamentals", "link": f"https://scholar.google.com/scholar?q={target_topic.replace(' ', '+')}"},
        {"type": "Practice Site", "title": f"Interactive {target_topic} Exercises", "link": "https://github.com"}
    ]

    rendered_md = f"### Curated Resources for: {target_topic}\n\n"
    for r in resources:
        rendered_md += f"- **[{r['type']}]** [{r['title']}]({r['link']})\n"

    return {
        "tool_name": "recommend_resources",
        "topic": target_topic,
        "resources": resources,
        "rendered_output": rendered_md
    }


# ==================== 14. PROGRESS TRACKER TOOL ====================

def track_progress_tool(completed_topic: str = "Topic Milestone", user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Log completed topics, update strong concepts in profile, and recalculate readiness."""
    target_topic = completed_topic or kwargs.get("topic", "Topic Milestone")
    profile = memory_manager.get_learner_profile(user_id)
    strong = list(profile.get("strong_concepts", []))
    weak = list(profile.get("weak_concepts", []))

    if target_topic not in strong:
        strong.append(target_topic)
    if target_topic in weak:
        weak.remove(target_topic)

    current_readiness = profile.get("readiness_score", 50.0)
    new_readiness = min(100.0, current_readiness + 5.0)

    memory_manager.update_profile(user_id=user_id, updates={
        "strong_concepts": strong,
        "weak_concepts": weak,
        "readiness_score": new_readiness
    })

    rendered_md = f"### Milestone Logged: {target_topic}\n\n"
    rendered_md += f"- Moved **{target_topic}** to Mastered Concepts! Check\n"
    rendered_md += f"- Updated Exam Readiness Score: **{new_readiness:.1f}%** (+5.0%)\n"

    return {
        "tool_name": "track_progress",
        "completed_topic": target_topic,
        "new_readiness": new_readiness,
        "rendered_output": rendered_md
    }


# ==================== 15. EXAM READINESS SCORE TOOL ====================

def calculate_exam_readiness_tool(user_id: str = "default_user", **kwargs) -> Dict[str, Any]:
    """Compute holistic exam readiness score based on quizzes, completed concepts, and study hours."""
    profile = memory_manager.get_learner_profile(user_id)
    quizzes = db.get_quiz_history(user_id=user_id)
    strong = profile.get("strong_concepts", [])
    weak = profile.get("weak_concepts", [])

    quiz_avg = (sum(q.get("percentage", 70.0) for q in quizzes) / len(quizzes)) if quizzes else 65.0
    concept_ratio = (len(strong) / max(1, len(strong) + len(weak))) * 100.0 if (strong or weak) else 50.0

    readiness = (0.6 * quiz_avg) + (0.4 * concept_ratio)
    readiness = max(10.0, min(98.0, readiness))

    memory_manager.update_profile(user_id=user_id, updates={"readiness_score": readiness})

    rendered_md = f"### Holistic Exam Readiness Score\n\n"
    rendered_md += f"## **{readiness:.1f}% / 100%**\n\n"
    rendered_md += f"- **Quiz Score Average**: {quiz_avg:.1f}%\n"
    rendered_md += f"- **Mastered Topic Ratio**: {concept_ratio:.1f}%\n"
    rendered_md += f"- **Status**: {'High Confidence' if readiness > 75 else 'Moderate Progress - Keep Revising'}\n"

    return {
        "tool_name": "calculate_exam_readiness",
        "readiness_score": readiness,
        "quiz_avg": quiz_avg,
        "rendered_output": rendered_md
    }


# ==================== MASTER DISPATCHER ====================

TOOL_REGISTRY = {
    "generate_quiz": generate_quiz_tool,
    "generate_flashcards": generate_flashcards_tool,
    "create_revision_planner": create_revision_planner_tool,
    "analyze_weak_topics": analyze_weak_topics_tool,
    "generate_learning_path": generate_learning_path_tool,
    "calculate_study_time": calculate_study_time_tool,
    "summarize_pdf_notes": summarize_pdf_notes_tool,
    "generate_study_notes": generate_study_notes_tool,
    "generate_mind_map": generate_mind_map_tool,
    "explain_concept": explain_concept_tool,
    "generate_daily_goals": generate_daily_goals_tool,
    "generate_motivation": generate_motivation_tool,
    "recommend_resources": recommend_resources_tool,
    "track_progress": track_progress_tool,
    "calculate_exam_readiness": calculate_exam_readiness_tool,
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any], user_id: str = "default_user") -> Dict[str, Any]:
    """Execute target tool by name with arguments and user_id parameter."""
    if tool_name not in TOOL_REGISTRY:
        return {
            "tool_name": tool_name,
            "status": "error",
            "error_message": f"Tool '{tool_name}' is not registered.",
            "rendered_output": f"Error: Tool '{tool_name}' not found."
        }

    tool_func = TOOL_REGISTRY[tool_name]
    tool_args_copy = dict(tool_args or {})
    tool_args_copy["user_id"] = user_id

    try:
        res = tool_func(**tool_args_copy)
        res["status"] = "success"
        return res
    except Exception as e:
        return {
            "tool_name": tool_name,
            "status": "error",
            "error_message": str(e),
            "rendered_output": f"Error executing {tool_name}: {str(e)}"
        }
