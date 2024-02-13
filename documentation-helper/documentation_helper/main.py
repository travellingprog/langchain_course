from typing import Set

import streamlit as st
from backend.core import run_llm
from streamlit_chat import message


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""

    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


def build_page():
    st.header("LangChain Udemy Course - Documentation Helper Bot")

    prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []

    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []

    if prompt:
        with st.spinner("Generating response..."):
            generated_response = run_llm(prompt)
            sources = set(
                [
                    doc.metadata["source"]
                    for doc in generated_response["source_documents"]
                ]
            )
            formatted_response = (
                f"{generated_response['result']}\n\n{create_sources_string(sources)}"
            )

            st.session_state["user_prompt_history"].append(prompt)
            st.session_state["chat_answers_history"].append(formatted_response)

    if st.session_state["chat_answers_history"]:
        for user_query, answer in zip(
            st.session_state["user_prompt_history"],
            st.session_state["chat_answers_history"],
        ):
            message(user_query, is_user=True)
            message(answer)


build_page()
