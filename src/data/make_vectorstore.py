from pathlib import Path
from typing import List

import click
import polars as pl
import structlog
from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PolarsDataFrameLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings

LOG = structlog.get_logger()


def load_processed_data() -> List[Document]:
    processed_data = pl.read_parquet("data/processed/processed_data.parquet")
    loader = PolarsDataFrameLoader(processed_data, page_content_column="page_content")
    documents = loader.load()
    LOG.debug("Loaded documents", first=documents[0], n=len(documents))
    return documents


def filter_documents(documents: List[Document]) -> List[Document]:
    filtered_documents = filter_complex_metadata(documents)
    LOG.debug(
        "Filtered documents", first=filtered_documents[0], n=len(filtered_documents)
    )
    return filtered_documents


def load_documents(filtered_documents: List[Document], output_filepath: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(
        filtered_documents, embeddings, persist_directory=output_filepath
    )
    LOG.info("Created vectorstore", vectorstore=vectorstore)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    LOG.info("Creating vectorstore", input_filepath=input_filepath)

    documents = load_processed_data()
    filtered_documents = filter_documents(documents)
    load_documents(filtered_documents, output_filepath)


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
