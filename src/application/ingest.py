import logging
from typing import Union

import polars as pl
import structlog

# Get a logger.
log = structlog.get_logger()


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_parquet(path)
    log.info("Read data", data=data.describe())
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

    return x_filtered


def collapse_sets(df) -> pl.DataFrame:
    return (
        df.group_by(
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


if __name__ == "__main__":
    # set log to info
    logging.basicConfig(level=logging.DEBUG)
    data = read_data("data/raw/cards.parquet")

    processed_data = data.lazy().pipe(filter_main_sets).pipe(collapse_sets).collect()

    print(processed_data)
