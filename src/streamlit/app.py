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


@st.cache_data
def generate_deck_theme_suggestion(commander_data: pd.DataFrame) -> str:
    """Generate deck theme suggestion based on the commander's abilities and general strategy of the deck."""
    deck_theme_prompt = PromptTemplate.from_template(
        """"
        Suggest a theme for a commander deck with {commander} as the commander.

        Do not suggest the commander card itself.

        Do not include any extra information in the response.

        Keep your theme description short and focussed on the commander's abilities.
        """
    )
    ic(deck_theme_prompt)

    theme_chain = deck_theme_prompt | LLM

    commander_description = commander_data.to_dict(orient="records")

    theme_result = theme_chain.invoke({"commander": commander_description})

    return theme_result


@st.cache_data
def generate_another_theme_suggestion(
    commander_data: pd.DataFrame, previous_theme
) -> str:
    """Generate another deck theme suggestion based on the commander's abilities and general strategy of the deck."""
    deck_theme_prompt = PromptTemplate.from_template(
        """"
        Suggest a theme for a commander deck with {commander} as the commander.

        Do not suggest the commander card itself.

        Do not include any extra information in the response.

        Keep your theme description short and focussed on the commander's abilities.

        Make a suggestion that is different from the previous theme.

        The previous theme was: {previous_theme}
        """
    )

    # TODO: Make the theme suggestion even more different from the previous theme

    theme_chain = deck_theme_prompt | LLM

    commander_description = commander_data.to_dict(orient="records")

    theme_result = theme_chain.invoke(
        {"commander": commander_description, "previous_theme": previous_theme}
    )

    return theme_result


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
    prompt = prompt.partial(commander=commander)

    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm=LLM, base_embeddings=EMBEDDINGS, custom_prompt=prompt
    )

    hyde_result = hyde_embeddings.embed_query(
        theme
    )  # TODO: report possible bug that hyde_embeddings.embed_query({"commander": commander, "theme": theme}) complains about key error, again. I got around this last time but having a prompt with 1 arg left

    hyde_search = VECTORSTORE.similarity_search_by_vector_with_relevance_scores(
        hyde_result, k=20
    )

    suggestions = [document[0].metadata for document in hyde_search]

    # log if commander card name in metadata
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


st.session_state.commander_name = str(st.selectbox("Commander", COMMANDER_NAMES))

st.session_state.commander_data = get_commander_data(
    commander=str(st.session_state.commander_name)
)

st.write(st.session_state.commander_data)

st.session_state.selected_theme = None

if "themes_generated" not in st.session_state:
    st.session_state.themes_generated = False
    st.session_state.themes = {"a": None, "b": None}


def click_theme_button():
    st.session_state.themes_generated = True


st.button("Generate deck theme suggestions", on_click=click_theme_button)

if st.session_state.themes_generated:
    st.session_state.themes = generate_deck_theme_suggestion(
        st.session_state.commander_data
    )

    st.session_state.themes = {
        "a": generate_deck_theme_suggestion(st.session_state.commander_data),
        "b": generate_another_theme_suggestion(
            st.session_state.commander_data, st.session_state.themes
        ),
    }

    st.session_state.selected_theme = st.radio(
        "Choose a deck theme suggestion",
        options=[st.session_state.themes["a"], st.session_state.themes["b"]],
    )

    if st.button("Generate card suggestions"):
        st.session_state.suggestions = {
            card_type: generate_card_suggestions(
                prompt=prompt,
                commander_data=st.session_state.commander_data,
                commander=st.session_state.commander_name,
                theme=st.session_state.selected_theme,
            )
            for card_type, prompt in partial_prompts.items()
        }

        # TODO: Instead of displaying each suggestion data set based on the expected input card type, aggregate them, then split into the actual types
        # TODO: Instead of displaying the vectors based on similarity, have them filtered/ reranked by an llm

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
