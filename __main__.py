import torch
from bidict import bidict
import click
import pickle
import os
from collections.abc import Hashable
from src.IdeologyWatch.datahandler import DataHandler
from src.IdeologyWatch.classifier import IdeologyClassifier


def load_or_preprocess() -> (
    tuple[
        dict[str, torch.Tensor], tuple[bidict[int, Hashable], ...], IdeologyClassifier
    ]
):
    if os.path.exists("model_inputs.pickle") and os.path.exists(
        "index_to_label.pickle"
    ):
        return load_preprocessed()
    else:
        return preprocess()


def preprocess() -> (
    tuple[
        dict[str, torch.Tensor], tuple[bidict[int, Hashable], ...], IdeologyClassifier
    ]
):
    # get data handler
    handler = DataHandler(
        "data/speeches.feather", "data/politicians.feather", "data/factions.feather"
    )

    # preprocess
    preprocessed_data = handler.preprocess_data(10)
    nspeaker = preprocessed_data["politician_id"].nunique()
    nparty = preprocessed_data["faction_id"].nunique()

    # get classifier
    classifier = IdeologyClassifier(num_speakers=nspeaker, num_parties=nparty)

    # get model inputs and index to label
    model_inputs, index_to_label = classifier.get_model_inputs(
        *(i for _, i in preprocessed_data.items())
    )

    # export data
    for file_name, content in zip(
        ("model_inputs", "index_to_label"), (model_inputs, index_to_label)
    ):
        with open(f"{file_name}.pickle", "wb") as f:
            pickle.dump(content, f)

    return model_inputs, index_to_label, classifier


def load_preprocessed() -> (
    tuple[
        dict[str, torch.Tensor], tuple[bidict[int, Hashable], ...], IdeologyClassifier
    ]
):
    # load preprocessed data
    with open("model_inputs.pickle", "rb") as f:
        model_inputs = pickle.load(f)
    with open("index_to_label.pickle", "rb") as f:
        index_to_label = pickle.load(f)

    nspeaker = len(index_to_label[0])
    nparty = len(index_to_label[1])

    classifier = IdeologyClassifier(num_speakers=nspeaker, num_parties=nparty)

    return model_inputs, index_to_label, classifier


@click.group()
def cli():
    pass


@cli.command(help="Resume training from checkpoint.")
@click.option(
    "--export-checkpoints",
    default=None,
    required=False,
    flag_value=60,
    help="Export checkpoints after specified number of minutes.",
)
def resume(export_checkpoints):
    model_inputs, index_to_label, classifier = load_or_preprocess()
    classifier.resume_training(
        model_input=model_inputs,
        batch_size=32,
        export_checkpoints=export_checkpoints,
        export_complete=True,
    )


@cli.command(help="Start training.")
@click.option(
    "--export-checkpoints",
    default=None,
    required=False,
    flag_value=60,
    help="Export checkpoints after specified number of minutes.",
)
def start(export_checkpoints):
    model_inputs, index_to_label, classifier = load_or_preprocess()
    classifier.start_training(
        model_input=model_inputs,
        # model_input=model_input,
        batch_size=32,
        export_checkpoints=export_checkpoints,
        export_complete=True,
    )


if __name__ == "__main__":
    cli()
