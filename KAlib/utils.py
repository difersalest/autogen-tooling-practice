import streamlit as st
from typing import List, Dict, Any, Optional

def display_session_msg(container_obj, user_image: Optional[str] = None):
    # Initialize messages list if not present
    messages = st.session_state.setdefault("messages", [])

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        avatar = None

        # Determine avatar to use
        if role == "user":
            avatar = user_image
        elif role not in ["user", "assistant"]:
            avatar = msg.get("image", None)

        # Display message
        if avatar:
            container_obj.chat_message(role, avatar=avatar).markdown(content)
        else:
            container_obj.chat_message(role).markdown(content)

def show_chat_history(container_obj, chat_history: List[Dict[str, Any]], user_image=None):
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for entry in chat_history:
        role = entry.get('role', 'user')
        content = entry.get('content', '')

        st.session_state.messages.append({"role": role, "content": content})

        # Display message if not empty
        if content.strip():
            if 'ALL DONE' in content:
                return  # Early exit on trigger keyword
            else:
                if role == 'assistant':
                    container_obj.chat_message("assistant", avatar=user_image).write(content)
                else:
                    container_obj.chat_message("user").write(content)

