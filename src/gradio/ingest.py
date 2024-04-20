import logging
from typing import Union

import polars as pl
import structlog
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PolarsDataFrameLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# Get a logger.
log = structlog.get_logger()


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_parquet(path)
    log.debug("Read data", data=data.describe())
    return data


def filter_main_sets(
    x: Union[pl.DataFrame, pl.LazyFrame],
) -> Union[pl.DataFrame, pl.LazyFrame]:
    recent_main_sets = {
        "ELD": "Throne of Eldraine",
        "THB": "Theros Beyond Death",
        "IKO": "Ikoria: Lair of Behemoths",
        "C20": "Commander 2020",
        "M21": "Core Set 2021",
        "ZNR": "Zendikar Rising",
        "ZNC": "Zendikar Rising Commander Decks",
        "CMR": "Commander Legends",
        "KHM": "Kaldheim",
        "KHC": "Kaldheim Commander Decks",
        "STX": "Strixhaven: School of Mages",
        "C21": "Commander 2021",
        "AFR": "Dungeons & Dragons: Adventures in the Forgotten Realms",
        "AFC": "Dungeons & Dragons: Adventures in the Forgotten Realms Commander Decks",
        "MID": "Innistrad: Midnight Hunt",
        "MIC": "Innistrad: Midnight Hunt Commander Decks",
        "VOW": "Innistrad: Crimson Vow",
        "VOC": "Innistrad: Crimson Vow Commander Decks",
        "NEO": "Kamigawa: Neon Dynasty",
        "NEC": "Kamigawa: Neon Dynasty Commander Decks",
        "SNC": "Streets of New Capenna",
        "NCC": "Streets of New Capenna Commander Decks",
        "CLB": "Commander Legends: Battle for Baldur's Gate",
        "DMU": "Dominaria United",
        "DMC": "Dominaria United Commander Decks",
        "UNF": "Unfinity",
        "40K": "Warhammer 40,000 Commander Decks",
        "BRO": "The Brothers' War",
        "BRC": "The Brothers' War Commander Decks",
        "ONE": "Phyrexia: All Will Be One",
        "ONC": "Phyrexia: All Will Be One Commander Decks",
        "MOM": "March of the Machine",
        "MOC": "March of the Machine/Commander decks",
        "LTR": "The Lord of the Rings: Tales of Middle-Earth",
        "LTC": "The Lord of the Rings: Tales of Middle-Earth Commander Decks",
        "WOE": "Wilds of Eldraine",
        "WOC": "Wilds of Eldraine/Commander decks",
        "LCI": "The Lost Caverns of Ixalan",
        "LCC": "The Lost Caverns of Ixalan/Commander decks",
        "MKM": "Murders at Karlov Manor",
        "MKC": "Murders at Karlov Manor/Commander decks",
    }
    x_filtered = x.filter(pl.col("setCode").is_in(list(recent_main_sets.keys())))

    log.debug("Filtered data", data=x_filtered.describe())
    return x_filtered


def collapse_sets(
    x: Union[pl.DataFrame, pl.LazyFrame],
) -> Union[pl.DataFrame, pl.LazyFrame]:
    x_collapsed = (
        x.group_by(
            "name",
            "colorIdentity",
            "colors",
            "faceConvertedManaCost",
            "keywords",
            "loyalty",
            "manaCost",
            "manaValue",
            "power",
            "subtypes",
            "supertypes",
            "toughness",
            "text",
            "type",
            "types",
        )
        .agg(pl.col("setCode").map_elements(list).alias("setCodes"))
        .sort("name")
    )
    log.debug(
        "Collapsed data", data=x_collapsed.describe(), columns=x_collapsed.columns
    )
    return x_collapsed


def create_page_content(
    x: Union[pl.DataFrame, pl.LazyFrame],
) -> Union[pl.DataFrame, pl.LazyFrame]:
    with_page_contents = x.with_columns(
        page_content=pl.col("type")
        + pl.col("text").fill_null("")
        + pl.col("keywords").fill_null(""),
    )
    log.debug(
        "Created page content",
        data=with_page_contents.describe(),
        columns=with_page_contents.columns,
    )
    return with_page_contents


def create_commander_selections(
    x: Union[pl.DataFrame, pl.LazyFrame],
) -> Union[pl.DataFrame, pl.LazyFrame]:
    # name does not start with "A-"
    commander_selections = x.filter(
        (pl.col("types") == "Creature")
        & (pl.col("supertypes") == "Legendary")
        & ~pl.col("name").str.starts_with("A-")  # untested
    )
    log.debug(
        "Commander data",
        data=commander_selections.describe(),
        columns=commander_selections.columns,
    )
    return commander_selections


if __name__ == "__main__":
    # set log to info
    logging.basicConfig(level=logging.INFO)
    data = read_data("data/raw/cards.parquet")

    processed_data = (
        data.lazy()
        .pipe(filter_main_sets)
        .pipe(collapse_sets)
        .pipe(create_page_content)
        .collect()
    )

    log.info(
        "Processed data",
        data=processed_data.describe(),
        columns=processed_data.columns,
    )

    commanders = create_commander_selections(processed_data)

    commanders.write_parquet("src/application/commanders.parquet")

    log.info("Wrote commanders", data=commanders.describe())

    loader = PolarsDataFrameLoader(processed_data, page_content_column="page_content")

    documents = loader.load()

    log.info("Loaded documents", first=documents[0], n=len(documents))

    filtered_documents = filter_complex_metadata(documents)

    log.info(
        "Filtered documents", first=filtered_documents[0], n=len(filtered_documents)
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma.from_documents(
        filtered_documents, embeddings, persist_directory="src/application/chroma"
    )

    log.info("Created vectorstore", vectorstore=vectorstore)
