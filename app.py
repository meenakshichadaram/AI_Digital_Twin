"""
AI Digital Twin of Knowledge - Main Streamlit SaaS Application

Entry point for the personalized learning companion platform.
Assembles custom CSS, top header navbar, sidebar router, and page views.
"""

import streamlit as st
from pathlib import Path

from config import settings
from components.navbar import render_navbar
from components.sidebar import render_sidebar

# Page Imports
from pages.landing import render_landing_page
from pages.chat import render_chat_page
from pages.memory_dashboard import render_memory_dashboard
from pages.knowledge_base import render_knowledge_base_page
from pages.quiz_center import render_quiz_center_page
from pages.revision_planner import render_revision_planner_page
from pages.flashcards import render_flashcards_page
from pages.settings import render_settings_page


# Page Configuration
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Glassmorphic CSS Theme
css_path = settings.CSS_DIR / "glassmorphic.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    """Main application layout and view router."""
    user_id = "default_user"

    # Render Sticky Top Header Bar
    render_navbar(user_id=user_id)

    # Render Navigation Sidebar
    selected_page = render_sidebar()

    # Route Selected Page View
    if selected_page == "Landing":
        render_landing_page(user_id=user_id)
    elif selected_page == "Chat":
        render_chat_page(user_id=user_id)
    elif selected_page == "Memory Dashboard":
        render_memory_dashboard(user_id=user_id)
    elif selected_page == "Knowledge Base":
        render_knowledge_base_page(user_id=user_id)
    elif selected_page == "Analytics":
        # Handled in Phase 8 analytics module
        try:
            from pages.analytics import render_analytics_page
            render_analytics_page(user_id=user_id)
        except ImportError:
            st.info("📊 Learning Analytics Module loading...")
    elif selected_page == "Quiz Center":
        render_quiz_center_page(user_id=user_id)
    elif selected_page == "Revision Planner":
        render_revision_planner_page(user_id=user_id)
    elif selected_page == "Flashcards":
        render_flashcards_page(user_id=user_id)
    elif selected_page == "Settings":
        render_settings_page(user_id=user_id)
    else:
        render_landing_page(user_id=user_id)


if __name__ == "__main__":
    main()
