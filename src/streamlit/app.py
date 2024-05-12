import pandas as pd
import polars as pl
import streamlit as st
import structlog
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.globals import set_debug
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings

set_debug(True)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

VECTORSTORE = Chroma(
    persist_directory="src/gradio/chroma",
)
COMMANDERS = pl.read_parquet("src/streamlit/commanders.parquet")

COMMANDER_NAMES = COMMANDERS["name"].to_list()

LOG = structlog.get_logger()

LLM = OpenAI()


def get_commander_data(commander: str) -> pd.DataFrame:
    return (
        COMMANDERS.filter(pl.col("name") == commander)
        .select(
            [
                "name",
                "colorIdentity",
                "manaCost",
                "types",
                "subtypes",
                "text",
                "power",
                "toughness",
            ]
        )
        .to_pandas()
    )


def generate_deck_theme_suggestions(commander_data: pd.DataFrame) -> str:
    """Generate deck theme suggestions based on the commander's abilities and general strategy of the deck."""
    deck_theme_prompt = PromptTemplate.from_template(
        """"
        Suggest a theme for a commander deck with {commander} as the commander. 
        Your suggestion should be based on the commander's abilities and the general strategy of the deck.

        Do not suggest the commander card itself.

        Do not include any extra information in the response.
        """
    )

    theme_chain = deck_theme_prompt | LLM

    commander_description = commander_data.to_dict(orient="records")

    theme_result = theme_chain.invoke({"commander": commander_description})

    return theme_result


creature_prompt = PromptTemplate.from_template(
    """"
    Design a single Magic the Gathering creature type card that would be valuable to include in a commander deck with {commander} as the commander. 
    Do not suggest the commander card itself.
    Your suggestions should be based on the commander's abilities and the general strategy of the deck.

    Your response should be a card formated in json, each with the following information:
    - Name
    - Mana cost
    - Supertypes
    - Types
    - Subtypes
    - Text
    - Power
    - Toughness

    Do not suggest to include the commander itself.

    Do not include any extra information in the response.

    Only use the commanders colour identity for the card.
    """
)

mtg_prompt_template = PromptTemplate.from_template(
    """"
    Design a single Magic the Gathering {type} type card that would be valuable to include in a commander deck with {commander} as the commander.
    The theme of the deck is {theme}. 
    Do not suggest the commander card itself.
    Your suggestions should be based on the commander's abilities and the general strategy of the deck.

    Your response should be a card formated in json, each with the following keys: {card_keys}

    Do not suggest to include the commander itself.

    Do not include any extra information in the response.

    Only use the commanders colour identity for the card.
    """
)

partial_prompts = {
    "creatures": mtg_prompt_template.partial(
        type="creature",
        card_keys="Name, Mana cost, Types, Subtypes, Text, Power, Toughness",
    ),
    "artifacts": mtg_prompt_template.partial(
        type="artifact", card_keys="Name, Mana cost, Types, Subtypes, Text"
    ),
    "instants": mtg_prompt_template.partial(
        type="instant", card_keys="Name, Mana cost, Types, Subtypes, Text"
    ),
    "enchantments": mtg_prompt_template.partial(
        type="enchantment", card_keys="Name, Mana cost, Types, Subtypes, Text"
    ),
    "sorceries": mtg_prompt_template.partial(
        type="sorcery", card_keys="Name, Mana cost, Types, Subtypes, Text"
    ),
}


def generate_card_suggestions(
    prompt: BasePromptTemplate, commander_data: pl.DataFrame, commander: str, theme: str
) -> pd.DataFrame:
    # use ic for each argument
    for arg in [prompt, commander_data, commander, theme]:
        ic(arg)

    prompt = prompt.partial(commander=commander)
    ic(prompt)

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=LLM, base_embeddings=EMBEDDINGS, custom_prompt=prompt
    )
    ic(hyde_embeddings)

    hyde_result = hyde_embeddings.embed_query(
        theme
    )  # BUG: hyde_embeddings.embed_query({"commander": commander, "theme": theme}) complains about key error, again. I got around this last time but having a prompt with 1 arg left

    hyde_search = VECTORSTORE.similarity_search_by_vector_with_relevance_scores(
        hyde_result, k=20
    )

    suggestions = [document[0].metadata for document in hyde_search]

    # log if commander card name in metadata
    # TODO: restructure the data prep into a separate function
    for item in suggestions:
        if commander in item["name"]:
            LOG.info("Commander card in metadata", metadata=item)

    commander_identity = commander_data["colorIdentity"].to_list()[0]

    def in_commander_colours(colors, color_list=commander_identity):
        return set(colors).issubset(set(color_list))

    possible_suggestion_column_names = [
        "name",
        "manaCost",
        "types",
        "subtypes",
        "text",
        "power",
        "toughness",
        "in_commander_colours",
    ]

    display_suggestions = (
        pl.DataFrame(suggestions)
        .with_columns(
            pl.col("colorIdentity")
            .str.split(by=", ")
            .map_elements(in_commander_colours, return_dtype=pl.Boolean)
            .alias("in_commander_colours"),
        )
        .filter(
            ~pl.col("name").str.contains(commander),
        )
    )

    existing_columns = []
    for column in possible_suggestion_column_names:
        if column in display_suggestions.columns:
            existing_columns.append(column)

    display_suggestions = display_suggestions.select(existing_columns)

    return display_suggestions.to_pandas()


st.session_state.commander_name: str = st.selectbox("Commander", COMMANDER_NAMES)

st.session_state.commander_data = get_commander_data(
    commander=st.session_state.commander_name
)

st.write(st.session_state.commander_data)

if st.button("Generate deck theme suggestions"):
    st.session_state.deck_theme_suggestion_1 = generate_deck_theme_suggestions(
        st.session_state.commander_data
    )
    st.write(st.session_state.deck_theme_suggestion_1)

    st.session_state.deck_theme_suggestion_2 = generate_deck_theme_suggestions(
        st.session_state.commander_data
    )
    st.write(st.session_state.deck_theme_suggestion_2)

if st.button("Use deck theme suggestion 1"):
    st.session_state.deck_theme = st.session_state.deck_theme_suggestion_1

    st.write("## Choosen theme", st.session_state.deck_theme)

    st.session_state.suggestions = {
        card_type: generate_card_suggestions(
            prompt=prompt,
            commander_data=st.session_state.commander_data,
            commander=st.session_state.commander_name,
            theme=st.session_state.deck_theme,
        )
        for card_type, prompt in partial_prompts.items()
    }

    st.write(
        "## Creatures",
        st.session_state.suggestions.get("creatures"),
        "## Enchantments",
        st.session_state.suggestions.get("enchantments"),
        "## Artifacts",
        st.session_state.suggestions.get("artifacts"),
        "## Instants",
        st.session_state.suggestions.get("instants"),
        "## Sorceries",
        st.session_state.suggestions.get("sorceries"),
    )

elif st.button("Use deck theme suggestion 2"):
    st.session_state.deck_theme = st.session_state.deck_theme_suggestion_2

    st.write("## Choosen theme", st.session_state.deck_theme)

    st.session_state.suggestions = {
        card_type: generate_card_suggestions(
            prompt=prompt,
            commander_data=st.session_state.commander_data,
            commander=st.session_state.commander_name,
            theme=st.session_state.deck_theme,
        )
        for card_type, prompt in partial_prompts.items()
    }

    st.write(
        "## Creatures",
        st.session_state.suggestions.get("creatures"),
        "## Enchantments",
        st.session_state.suggestions.get("enchantments"),
        "## Artifacts",
        st.session_state.suggestions.get("artifacts"),
        "## Instants",
        st.session_state.suggestions.get("instants"),
        "## Sorceries",
        st.session_state.suggestions.get("sorceries"),
    )
