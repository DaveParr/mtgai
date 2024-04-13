import gradio as gr
import polars as pl
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import (
    PromptTemplate,
)
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    persist_directory="src/application/chroma",
)

commanders = pl.read_parquet("src/application/commanders.parquet")

commander_names = commanders["name"].to_list()


def card_suggestions(commander):
    # return the commander tha matches the name
    commander_data = commanders.filter(pl.col("name") == commander)
    print(commander_data)
    return commander_data.select(["name", "colorIdentity", "types", "subtypes"])


demo = gr.Interface(
    fn=card_suggestions,
    live=True,
    inputs=gr.Dropdown(
        commander_names, label="Comander", info="Select a commander for the deck"
    ),
    outputs=gr.Dataframe(),
)

llm = OpenAI()

# mtg_card_prompt = PromptTemplate.from_template(
#     "Suggest a Magic the Gathering card that would fit in a commander deck with {commander} as the commander"
# )

# hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
#     llm=llm, base_embeddings=embeddings, custom_prompt=mtg_card_prompt
# )

# hyde_result_urza = hyde_embeddings.embed_query("Urza, Lord Artificier")

# hyde_search_urza = vectorstore.similarity_search_by_vector_with_relevance_scores(
#     hyde_result_urza, k=10
# )

if __name__ == "__main__":
    demo.launch()
