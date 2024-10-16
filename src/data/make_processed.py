from datetime import date
from pathlib import Path

import click
import polars as pl
import structlog
from dotenv import find_dotenv, load_dotenv

LOG = structlog.get_logger()


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_parquet(path)
    LOG.debug("Read data", data=data.describe(), path=path)
    return data


def filter_main_sets(
    x: pl.DataFrame, sets: pl.DataFrame, cutoff_date: date
) -> pl.DataFrame:
    filtered_sets = sets.with_columns(
        pl.col("releaseDate").str.to_date("%Y-%m-%d").alias("releaseDate"),
        pl.col("code").alias("setCode"),
    ).filter(
        (pl.col("releaseDate") > cutoff_date)
        & (pl.col("isOnlineOnly") != True)  # noqa: E712
        & pl.col("type").is_in(["core", "expansion", "commander", "draft_innovation"])
    )

    LOG.debug(
        "Filtered data",
        data=filtered_sets.describe(),
        columns=filtered_sets.columns,
        n=filtered_sets.height,
    )
    return x.join(filtered_sets, on="setCode")


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


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option(
    "--cutoff_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=date.today(),
    help="Cutoff date for sets",
)  # TODO: Make this date setable via the cli call
def main(input_filepath, output_filepath, cutoff_date: date):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    print(date)

    cards_data = read_data("data/raw/cards.parquet")
    sets = read_data("data/raw/sets.parquet")

    processed_data: pl.DataFrame = (
        cards_data.pipe(filter_main_sets, sets=sets, cutoff_date=cutoff_date)
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


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
