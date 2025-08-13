import os
import asyncio
import streamlit as st
from convo import OrchestratedConversationalSystem, AgentState
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Session configuration
session = {
    "api_key": st.secrets.get("OPENAI_API_KEY", ""),
    "model_name": "gpt-4",
    "base_url": "https://api.openai.com/v1",
    "persona_name": "Calum",
    "avatar_id": "calum",
    "avatar_prompts": {
        "calum": "You are Calum Worthy, a witty activist and actor."
    },
    "temperature": 0.3,
    "debug": False,
    "force_sync_flush": False
}

# Initialize the embedding model BEFORE constructing the conversation system
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=session.get("api_key")
)

# --- Init session state ---
if "conv" not in st.session_state:
    st.session_state.conv = OrchestratedConversationalSystem(session=session)

if "state" not in st.session_state:
    st.session_state.state = AgentState(
        session=session,
        scratchpad=[],
        selected_context="",
        compressed_history="",
        agent_context="",
        response=""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []  # stores dicts: {"role": "user"/"assistant", "content": "..."}

# --- Title ---
st.title("🎭 Calum Worthy - Your Virtual Best Friend")

# --- Render past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input box ---
if prompt := st.chat_input("Type your message and press Enter..."):
    # 1. Show user message instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Calum is thinking..."):
            new_state = asyncio.run(st.session_state.conv.run_turn(prompt, st.session_state.state))
            reply = new_state.get("response", "")
            st.markdown(reply)

            # Save state + assistant message
            st.session_state.state.update(new_state)
            st.session_state.messages.append({"role": "assistant", "content": reply})

# --- End Chat button ---
if st.button("End Chat"):
    asyncio.run(st.session_state.conv.store.flush())
    st.session_state.messages.clear()
    st.session_state.state = AgentState(session=session, scratchpad=[])
    st.rerun()
