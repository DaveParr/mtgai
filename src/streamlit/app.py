import logging

import comm
import pandas as pd
import polars as pl
import streamlit as st
import structlog
from click import command
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.globals import set_debug
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

set_debug(True)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    persist_directory="src/gradio/chroma",
)
commanders = pl.read_parquet("src/streamlit/commanders.parquet")

commander_names = commanders["name"].to_list()

log = structlog.get_logger()

llm = OpenAI()


def get_commander_data(commander) -> pd.DataFrame:
    return (
        commanders.filter(pl.col("name") == commander)
        .select(
            ["name", "manaCost", "supertypes", "subtypes", "text", "power", "toughness"]
        )
        .to_pandas()
    )


def generate_card_suggestions(commander) -> pd.DataFrame:
    # return the commander tha matches the name
    commander_data = commanders.filter(pl.col("name") == commander)

    mtg_card_prompt = PromptTemplate.from_template(
        """"
        Design a Magic the Gathering card that would be valuable to include in a commander deck with {commander} as the commander. 
        Do not suggest the commander card itself.
        Your suggestions should be based on the commander's abilities and the general strategy of the deck.

        Your response should be a card formated in json, each with the following information:
        - Name
        - Mana cost
        - Supertypes
        - Subtypes
        - Text
        - Power
        - Toughness

        Do not include the commander card in the suggestions.

        Do not include any extra information in the response.
        """
    )

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm, base_embeddings=embeddings, custom_prompt=mtg_card_prompt
    )

    hyde_result = hyde_embeddings.embed_query(commander_data.to_dict(as_series=False))

    hyde_search = vectorstore.similarity_search_by_vector_with_relevance_scores(
        hyde_result, k=20
    )

    suggestions = [document[0].metadata for document in hyde_search]

    # log if commander card name in metadata
    for item in suggestions:
        if commander in item["name"]:
            log.info("Commander card in metadata", metadata=item)

    display_suggestions = (
        pl.DataFrame(suggestions)
        .select(
            ["name", "manaCost", "supertypes", "subtypes", "text", "power", "toughness"]
        )
        .filter(~pl.col("name").str.contains(commander))
    )

    return display_suggestions.to_pandas()


commander = st.selectbox("Commander", commander_names)

st.write(get_commander_data(commander))

suggestions = st.button("Get Card Suggestions")

if suggestions:
    st.write(generate_card_suggestions(commander))
