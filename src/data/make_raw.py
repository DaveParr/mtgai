# -*- coding: utf-8 -*-
from pathlib import Path
from zipfile import ZipFile

import click
import polars as pl
import requests
import structlog
from dotenv import find_dotenv, load_dotenv


def download_mtgjson_parquet(raw_directory, url):
    log = structlog.get_logger()

    log.info("Downloading zip file", url=url)
    response = requests.get(url)

    file_name_with_zip = url.split("/")[-1]
    file_name = file_name_with_zip.replace(".zip", "")

    raw_cards_filepath = raw_directory + "/" + file_name_with_zip

    log.info("Writing zip file", filepath=raw_cards_filepath)
    with open(raw_cards_filepath, "wb") as file:
        file.write(response.content)

    log.info("Reading parquet file", filepath=raw_cards_filepath)
    data = pl.read_parquet(ZipFile(raw_cards_filepath).read(file_name))

    data.describe()

    log.info("Saving parquet file", filepath=raw_directory)
    data.write_parquet(raw_directory + "/" + file_name)

    return data


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    download_mtgjson_parquet(
        input_filepath, url="https://mtgjson.com/api/v5/parquet/cards.parquet.zip"
    )

    download_mtgjson_parquet(
        input_filepath,
        url="https://mtgjson.com/api/v5/parquet/sets.parquet.zip",
    )


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
