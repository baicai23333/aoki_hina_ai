"""Streamlit entry point for the chat site and its protected admin console."""

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Aoki Hina AI",
    page_icon="🐈",
)

chat_page = st.Page(
    ROOT / "chat_client.py",
    title="聊天",
    icon=":material/chat:",
    default=True,
)
admin_page = st.Page(
    ROOT / "admin_page.py",
    title="管理后台",
    icon=":material/admin_panel_settings:",
    url_path="admin",
)

navigation = st.navigation(
    [chat_page, admin_page],
    position="hidden",
)
navigation.run()
