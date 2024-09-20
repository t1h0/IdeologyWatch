import pandas as pd
from pathlib import Path
from loguru import logger
from typing import NamedTuple


class DataHandler:

    class SamplesLabels(NamedTuple):
        speeches: list[str]
        politicians: list[str]
        factions: list[str]

    def __init__(
        self,
        speeches_path: str | Path,
        politicians_path: str | Path,
        factions_path: str | Path,
    ) -> None:
        self.speeches = pd.read_feather(speeches_path)
        self.politicians = pd.read_feather(politicians_path)
        self.factions = pd.read_feather(factions_path)

    def preprocess_data(self, min_speeches_per_speaker: int) -> pd.DataFrame:
        logger.info("Preprocessing data..")
        out = self.speeches[["speech_content", "politician_id", "faction_id"]]

        # remove unknown speakers and unknown factions
        out = out.loc[(out["politician_id"] != -1) & (out["faction_id"] != -1)]

        # remove speakers with less than min_speeches_per_speaker speeches
        out = out.merge(
            out["politician_id"]
            .value_counts()
            .loc[lambda x: x >= min_speeches_per_speaker]
            .reset_index()["politician_id"],
            how="inner",
        )

        # remove contributions
        out["speech_content"] = out["speech_content"].str.replace(
            r"\s*\(\{\d+\}\)\s*", " ", regex=True
        )

        return out
