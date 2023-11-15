import os
from urllib.parse import urlparse
from openai import OpenAI
import numpy as np
import pandas as pd
import argparse
import tiktoken

from utils.crawler import crawl
from keys import OPENAI_API_KEY


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def cosine_similarity(a, b):
    """Use to calculate cosine similarity of 2 vectors a and b"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_embedding(client, text, model="text-embedding-ada-002"):
    """Use to get embedding vector from text"""
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def search_docs(df, user_query, client, top_n=4):
    embedding = get_embedding(client, user_query, model="text-embedding-ada-002")
    df["similarities"] = df.ada_embedding.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    res = df.sort_values("similarities", ascending=False).head(top_n)
    return res


def load_text(url):
    # Create a list to store the text files
    texts = []

    domain = urlparse(url).netloc

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):
        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            texts.append(
                (file[:-9].replace(domain, "").replace("-", " ").replace("_", ""), text)
            )

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=["fname", "text"])

    # Set the text column to be the raw text with the newlines removed
    df["text"] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv("processed/scraped.csv")


def tokenize():
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv("processed/scraped.csv", index_col=0)
    df.columns = ["title", "text"]

    # Tokenize the text and save the number of tokens to a new column
    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df


def get_embedded_data(client, url):
    load_text(url)
    df = tokenize()
    df["ada_embedding"] = df.text.apply(
        lambda x: get_embedding(client, x, model="text-embedding-ada-002")
    )
    df.to_csv("processed/embeddings.csv")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Enter URL to crawl and embedding data"
    )
    parser.add_argument(
        "--url", type=str, required=True, help="The URL needed to extract information"
    )

    args = parser.parse_args()
    url = args.url

    api_key = OPENAI_API_KEY
    client = OpenAI(api_key=api_key)

    print(f"[INFO] Crawling and embedding data from {url}")
    crawl(url)
    get_embedded_data(client, url)
    print(f"Data is saved at /processed/embeddings.csv")


if __name__ == "__main__":
    main()
