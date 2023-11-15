import numpy as np


def cosine_similarity(a, b):
    """Use to calculate cosine similarity of 2 vectors a and b"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_docs(df, user_query, client, top_n=4):
    embedding = get_embedding(client, user_query, model="text-embedding-ada-002")
    df["similarities"] = df.ada_embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    res = df.sort_values("similarities", ascending=False).head(top_n)
    return res


def get_embedding(client, text, model="text-embedding-ada-002"):
    """Use to get embedding vector from text"""
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def create_context(client, question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = (
        client.embeddings.create(input=question, model="text-embedding-ada-002")
        .data[0]
        .embedding
    )

    # Get the similarities from the embeddings
    df["similarities"] = df.ada_embedding.apply(
        lambda x: cosine_similarity(x, q_embeddings)
    )

    returns = []
    cur_len = 0

    for i, row in df.sort_values("similarities", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
    client,
    df,
    model="gpt-3.5-turbo",
    question="What is consent?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None,
    stream=True,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        client,
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n",
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            stream=stream,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""
