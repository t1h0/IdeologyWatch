from nomopytools.ml.nlp import MultiLabelSequenceClassifier


class IdeologyClassifier(MultiLabelSequenceClassifier):

    def __init__(
        self,
        num_speakers: int,
        num_parties: int,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            model_name="xlm-roberta-base",
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            freeze_model=True,
            random_seed=42,
            heads={"speaker": num_speakers, "party": num_parties},
            *args,
            **kwargs
        )
