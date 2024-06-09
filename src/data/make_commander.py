import logging
from typing import Union

import polars as pl
import structlog
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PolarsDataFrameLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# Get a logger.
LOG = structlog.get_logger()


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_parquet(path)
    LOG.debug("Read data", data=data.describe())
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
        "PIP": "Fallout",
        "OTJ": "Outlaws",
    }
    x_filtered = x.filter(pl.col("setCode").is_in(list(recent_main_sets.keys())))

    LOG.debug("Filtered data", data=x_filtered.describe())
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
    LOG.debug(
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
    LOG.debug(
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
    LOG.debug(
        "Commander data",
        data=commander_selections.describe(),
        columns=commander_selections.columns,
    )
    return commander_selections


if __name__ == "__main__":
    # set log to info
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    data = read_data("data/raw/cards.parquet")

    processed_data = (
        data.lazy()
        .pipe(filter_main_sets)
        .pipe(collapse_sets)
        .pipe(create_page_content)
        .collect()
    )

    LOG.info(
        "Processed data",
        data=processed_data.describe(),
        columns=processed_data.columns,
    )

    processed_data.write_parquet("data/processed/processed_data.parquet")

    commanders = create_commander_selections(processed_data)

    commanders.write_parquet("data/processed/commanders.parquet")

    LOG.info("Wrote commanders", data=commanders.describe())

    loader = PolarsDataFrameLoader(processed_data, page_content_column="page_content")

    documents = loader.load()

    LOG.info("Loaded documents", first=documents[0], n=len(documents))

    filtered_documents = filter_complex_metadata(documents)

    LOG.info(
        "Filtered documents", first=filtered_documents[0], n=len(filtered_documents)
    )

    # TODO: break into `make vectorstore` command
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # vectorstore = Chroma.from_documents(
    #     filtered_documents, embeddings, persist_directory="data/vectorstore"
    # )
    # LOG.info("Created vectorstore", vectorstore=vectorstore)
