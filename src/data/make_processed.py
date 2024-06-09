import logging
from datetime import date

import polars as pl
import structlog
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PolarsDataFrameLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import column

# Get a logger.
LOG = structlog.get_logger()


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_parquet(path)
    LOG.debug("Read data", data=data.describe())
    return data


def filter_main_sets(
    x: pl.DataFrame, sets: pl.DataFrame, cutoff_date: date
) -> pl.DataFrame:
    filtered_sets = (
        sets.with_columns(
            pl.col("releaseDate").str.to_date("%Y-%m-%d").alias("releaseDate"),
            pl.col("code").alias("setCode"),
        )
        .filter(
            (pl.col("releaseDate") > cutoff_date)
            & (pl.col("isOnlineOnly") != True)
            & pl.col("type").is_in(
                ["core", "expansion", "commander", "draft_innovation"]
            )
        )
        .join(x, on="setCode", validate="1:m")
    )

    LOG.debug(
        "Filtered data", data=filtered_sets.describe(), columns=filtered_sets.columns
    )
    return filtered_sets


def collapse_sets(
    x: pl.DataFrame,
) -> pl.DataFrame:
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
    x: pl.DataFrame,
) -> pl.DataFrame:
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
    x: pl.DataFrame,
) -> pl.DataFrame:
    commander_selections = x.filter(
        (pl.col("types") == "Creature")
        & (pl.col("supertypes") == "Legendary")
        & ~pl.col("name").str.starts_with(
            "A-"
        )  # name does not start with "A-" because might be 'alchemy' cards?
    )
    LOG.debug(
        "Commander data",
        data=commander_selections.describe(),
        columns=commander_selections.columns,
    )
    return commander_selections


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cards_data = read_data("data/raw/cards.parquet")
    sets = read_data("data/raw/sets.parquet")

    processed_data: pl.DataFrame = (
        cards_data.pipe(filter_main_sets, sets=sets, cutoff_date=date(2023, 1, 1))
        .pipe(collapse_sets)
        .pipe(create_page_content)
    )

    LOG.info(
        "Processed data",
        data=processed_data.describe(),
        columns=processed_data.columns,
    )

    processed_data.write_parquet("data/processed/processed_data.parquet")

    commanders: pl.DataFrame = create_commander_selections(processed_data)

    commanders.write_parquet("data/processed/commanders.parquet")

    LOG.info("Wrote commanders", data=commanders.describe())

    # # TODO: break into `make vectorstore` command
    loader = PolarsDataFrameLoader(processed_data, page_content_column="page_content")

    documents = loader.load()

    LOG.info("Loaded documents", first=documents[0], n=len(documents))

    filtered_documents = filter_complex_metadata(documents)

    LOG.info(
        "Filtered documents", first=filtered_documents[0], n=len(filtered_documents)
    )

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # vectorstore = Chroma.from_documents(
    #     filtered_documents, embeddings, persist_directory="data/vectorstore"
    # )
    # LOG.info("Created vectorstore", vectorstore=vectorstore)
