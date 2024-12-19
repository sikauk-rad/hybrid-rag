from tiktoken import get_encoding
from typing import Literal
from beartype import beartype
import numpy as np
from numpy.typing import NDArray
from .base import TokeniserInterface
from transformers import AutoTokenizer


@beartype
class OpenAITokeniserInterface(TokeniserInterface):

    def __init__(
        self,
        encoding: Literal[
            'o200k_base', 
            'cl100k_base', 
            'p50k_base', 
            'r50k_base', 
            'p50k_edit', 
            'gpt2',
        ],
    ) -> None:

        self.tokeniser = get_encoding(encoding)


    def tokenise(
        self,
        text: str,
    ) -> list[int]:

        return self.tokeniser.encode(text)


    def tokenise_multiple(
        self,
        texts: list[str],
    ) -> NDArray[np.integer] | list[list[int]]:

        return self.tokeniser.encode_batch(texts)


    def get_token_length(
        self,
        text: str,
    ) -> int:

        return len(self.tokeniser.encode(text))


    def get_token_lengths(
        self,
        texts: list[str],
    ) -> list[int]:

        return [*map(len, self.tokeniser.encode_batch(texts))]


@beartype
class HuggingFaceTokeniserInterface(TokeniserInterface):

    def __init__(
        self,
        model_id: str,
        token: str | None,
    ) -> None:

        self.model_id = model_id
        self.tokeniser = AutoTokenizer.from_pretrained(
            model_id,
            token = token,
        )


    def tokenise(
        self,
        text: str,
    ) -> list[int]:

        return self.tokeniser.encode(text)


    def tokenise_multiple(
        self,
        texts: list[str],
    ) -> NDArray[np.integer] | list[list[int]]:

        return [*map(self.tokeniser.encode, texts)]


    def get_token_length(
        self,
        text: str,
    ) -> int:

        return len(self.tokeniser.encode(text))


    def get_token_lengths(
        self,
        texts: list[str],
    ) -> list[int]:

        return [len(self.tokeniser.encode(text)) for text in texts]