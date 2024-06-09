import pandas as pd
import polars as pl
import streamlit as st
import structlog
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.globals import set_debug
from langchain.prompts import BasePromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

set_debug(True)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

VECTORSTORE = Chroma(
    persist_directory="data/vectorstore",
)
COMMANDERS = pl.read_parquet("data/processed/commanders.parquet")

COMMANDER_NAMES = COMMANDERS["name"].to_list()

LOG = structlog.get_logger()


LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")


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


class DeckThemes(BaseModel):
    theme_name_1: str = Field(description="Name of the first theme")
    theme_description_1: str = Field(description="Description of the first theme")
    theme_mechanics_1: str = Field(description="Mechanics of the first theme")
    theme_name_2: str = Field(description="Name of the second theme")
    theme_description_2: str = Field(description="Description of the second theme")
    theme_mechanics_2: str = Field(description="Mechanics of the second theme")


@st.cache_data
def generate_deck_theme_suggestion(commander_data: pd.DataFrame) -> DeckThemes:
    """Generate deck theme suggestion based on the commander's abilities and general strategy of the deck."""
    deck_theme_prompt = PromptTemplate.from_template(
        """"
        Suggest two different themes for a commander deck with {commander} as the commander.

        Do not suggest the commander card itself.

        Do not include any extra information in the response.

        Keep your theme description short and focussed on the commander's abilities.

        Suggest two themes that are different from each other.

        Example:
        commander_input:{{
            "name": "Urza, Chief Artificer",
            "colorIdentity": "B, U, W",
            "manaCost": "{{3}}{{W}}{{U}}{{B}}",
            "types": "Creature",
            "subtypes": "Human, Artificer",
            "text": "Affinity for artifact creatures (This spell costs {{1}} less to cast for each artifact creature you control.)\nArtifact creatures you control have menace.\nAt the beginning of your end step, create a 0/0 colorless Construct artifact creature token with "This creature gets +1/+1 for each artifact you control.",
            "power": "4",
            "toughness": "5"
            }}
        theme_output:{{
            theme_name_1: "Artifact Affinity"
            theme_description_1: "Go wide with artifact creatures to maximise affinity and construct tokens power and toughness"
            theme_mechanics_1: "Affinity, Artifact token generation"
            theme_name_2: "Artifact Aggro"
            theme_description_2: "Focus on attacking with artifact creatures that benefit from gaining menance"
            theme_mechanics_2: "Menace, Artifact creature synergy"
        }}
        Example:
        commander_input:{{
            "name": "Krenko, Mob Boss",
            "colorIdentity": "R",
            "manaCost": "{{2}}{{R}}",
            "types": "Creature",
            "subtypes": "Goblin",
            "text": "Tap: Create X 1/1 red Goblin creature tokens, where X is the number of Goblins you control.",
            "power": "3",
            "toughness": "3"
            }}
        theme_output:{{
            theme_name_1: "Goblin Tribal"
            theme_description_1: "Create a swarm of goblin tokens and use them to overwhelm your opponents"
            theme_mechanics_1: "Token generation, Goblin synergy"
            theme_name_2: "Goblin Aggro"
            theme_description_2: "Focus on attacking with goblin tokens that benefit from gaining haste"
            theme_mechanics_2: "Haste, Goblin creature synergy"
        }}
        Example:
        commander_input:{{
            "name": "Atraxa, Praetors' Voice",
            "colorIdentity": "B, G, U, W",
            "manaCost": "{{2}}{{W}}{{U}}{{B}}{{G}}",
            "types": "Creature",
            "subtypes": "Angel, Horror, Praetor",
            "text": "Flying, vigilance, deathtouch, lifelink\nAt the beginning of your end step, proliferate.",
            "power": "4",
            "toughness": "4"
            }}
        theme_output:{{
            theme_name_1: "Superfriends"
            theme_description_1: "Focus on planeswalkers and proliferate to maximise their loyalty counters"
            theme_mechanics_1: "Proliferate, Planeswalker synergy"
            theme_name_2: "Voltron"
            theme_description_2: "Focus on attacking with Atraxa to maximise the number of counters on her"
            theme_mechanics_2: "Counter synergy, Voltron strategy"
        }}
        """
    )

    theme_structured_llm = LLM.with_structured_output(DeckThemes)

    theme_chain = deck_theme_prompt | theme_structured_llm

    commander_description = commander_data.to_dict(orient="records")

    theme_result = theme_chain.invoke({"commander": commander_description})

    if type(theme_result) is not DeckThemes:
        raise ValueError("Invalid response from theme generation")

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
    "lands": mtg_prompt_template.partial(
        type="land", card_keys="Name, Types, Subtypes, Text"
    ),
    "planeswalkers": mtg_prompt_template.partial(
        type="planeswalker", card_keys="Name, Mana cost, Types, Subtypes, Text, Loyalty"
    ),
    "sagas": mtg_prompt_template.partial(
        type="saga", card_keys="Name, Mana cost, Types, Subtypes, Text"
    ),
}


def generate_card_suggestions(
    prompt: BasePromptTemplate, commander_data: pd.DataFrame, commander: str, theme: str
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

    display_suggestions: pl.DataFrame = (
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


def combine_in_colour_suggestions(suggestions: dict) -> pd.DataFrame:
    df = pd.concat(suggestions.values()).drop_duplicates()

    return df[df.in_commander_colours]


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
    ).dict()

    st.session_state.selected_theme = st.radio(
        "Choose a deck theme suggestion",
        options=[
            st.session_state.themes["theme_name_1"]
            + " - "
            + st.session_state.themes["theme_description_1"]
            + " - "
            + st.session_state.themes["theme_mechanics_1"],
            st.session_state.themes["theme_name_2"]
            + " - "
            + st.session_state.themes["theme_description_2"]
            + " - "
            + st.session_state.themes["theme_mechanics_2"],
        ],
    )

    if st.button("Generate card suggestions") and st.session_state.selected_theme:
        st.session_state.suggestions = {
            card_type: generate_card_suggestions(
                prompt=prompt,
                commander_data=st.session_state.commander_data,
                commander=st.session_state.commander_name,
                theme=st.session_state.selected_theme,
            )
            for card_type, prompt in partial_prompts.items()
        }

        st.session_state.all_in_colour_suggestions = combine_in_colour_suggestions(
            st.session_state.suggestions
        )

        # TODO: Instead of displaying the vectors based on similarity, have them filtered/ reranked by an llm

        st.write(
            "## Suggestions",
            st.session_state.all_in_colour_suggestions,
        )
