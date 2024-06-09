# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from zipfile import ZipFile

import click
import polars as pl
import requests
import structlog
from dotenv import find_dotenv, load_dotenv


def download_cards_parquet(raw_directory):
    log = structlog.get_logger()

    url = "https://mtgjson.com/api/v5/parquet/cards.parquet.zip"

    log.info("Downloading zip file", url=url)
    response = requests.get(url)

    raw_cards_filepath = raw_directory + "/cards.parquet.zip"

    log.info("Writing zip file", filepath=raw_cards_filepath)
    with open(raw_cards_filepath, "wb") as file:
        file.write(response.content)

    log.info("Reading parquet file", filepath=raw_cards_filepath)
    cards = pl.read_parquet(ZipFile(raw_cards_filepath).read("cards.parquet"))

    cards.describe()

    log.info("Saving parquet file", filepath=raw_directory)
    cards.write_parquet(raw_directory + "/cards.parquet")

    return cards


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    download_cards_parquet(input_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
