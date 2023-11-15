import streamlit as st
import openai
from openai import OpenAI
from utils.crawler import crawl
from utils.retriever import create_context
import pandas as pd
import numpy as np
from keys import OPENAI_API_KEY


def main(client, df):
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({"role": "user", "content": question})

        context = create_context(client, question, df)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            full_response = ""

            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {
                        "role": "system",
                        "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n. Please provide the answer in the form of bullet points.",
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                    },
                ],
                temperature=0,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=True,
            ):
                response_chunk = response.choices[0].delta.content
                full_response += response_chunk if response_chunk else ""
                message_placeholder.markdown(full_response + "| ")

            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


@st.cache_resource
def get_data():
    df = pd.read_csv(r"processed\embeddings.csv")
    df["ada_embedding"] = df["ada_embedding"].apply(eval).apply(np.array)
    return df


if __name__ == "__main__":
    client = OpenAI(api_key=OPENAI_API_KEY)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    df = get_data()

    main(client, df)
