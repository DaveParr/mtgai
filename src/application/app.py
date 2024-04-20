import logging
from importlib import metadata

import gradio as gr
import polars as pl
import structlog
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.globals import set_debug
from langchain.prompts import (
    PromptTemplate,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

set_debug(True)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    persist_directory="src/application/chroma",
)

commanders = pl.read_parquet("src/application/commanders.parquet")

commander_names = commanders["name"].to_list()

log = structlog.get_logger()


def card_suggestions(commander) -> tuple[pl.DataFrame, pl.DataFrame]:
    # return the commander tha matches the name
    commander_data = commanders.filter(pl.col("name") == commander)

    mtg_card_prompt = PromptTemplate.from_template(
        "Suggest a Magic the Gathering card that would be valuable to include in a commander deck with {commander} as the commander. Do not suggest the commander card itself."
    )

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=llm, base_embeddings=embeddings, custom_prompt=mtg_card_prompt
    )

    hyde_result = hyde_embeddings.embed_query(commander_data.to_dict(as_series=False))

    hyde_search = vectorstore.similarity_search_by_vector_with_relevance_scores(
        hyde_result, k=10
    )

    metadata_list = [document[0].metadata for document in hyde_search]

    # log if commander card name in metadata
    for item in metadata_list:
        if commander in item["name"]:
            log.info("Commander card in metadata", metadata=item)

    display_commander = commander_data.select(["name", "manaCost", "type", "text"])

    display_metadata = (
        pl.DataFrame(metadata_list)
        .select(["name", "manaCost", "type", "text"])
        .filter(~pl.col("name").str.contains(commander))
    )

    return display_commander, display_metadata


demo = gr.Interface(
    fn=card_suggestions,
    inputs=gr.Dropdown(
        commander_names, label="Comander", info="Select a commander for the deck"
    ),
    outputs=[  # commander data
        gr.DataFrame(),
        # metadata
        gr.DataFrame(),
    ],
)

llm = OpenAI()


if __name__ == "__main__":
    demo.launch()
