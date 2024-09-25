from typing import Self
from collections.abc import Iterable
from abc import ABC, abstractmethod
from numbers import Number
from pathlib import Path
from sklearn.exceptions import NotFittedError


class TextTransformer(ABC):

    @abstractmethod
    def fit(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    async def afit(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    def fit_transform(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    async def afit_transform(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ):
        ...


    def _check_fit(
        self,
    ) -> None:

        if not self._is_fit:
            raise NotFittedError('fit or fit_transform must be called first.')


    @abstractmethod
    def transform(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        pass


    @abstractmethod
    async def atransform(
        self,
        text: str,
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    def transform_multiple(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    async def atransform_multiple(
        self,
        texts: list[str],
        *args,
        **kwargs,
    ):
        ...


    @abstractmethod
    def score(
        self,
        text: str,
        document_indices: Iterable[int] | None,
        documents: Iterable[Iterable[Number]] | None,
        *args,
        **kwargs,
    ) -> Iterable[Number]:
        ...


    @abstractmethod
    async def ascore(
        self,
        text: str,
        document_indices: Iterable[int] | None,
        documents: Iterable[Iterable[Number]] | None,
        *args,
        **kwargs,
    ) -> Iterable[Number]:
        ...


    @classmethod
    @abstractmethod
    def load_from_file(
        cls,
        file_path: Path,
        *args,
        **kwargs,
    ) -> Self:
        ...


    @abstractmethod
    def save_to_file(
        self,
        save_path: Path,
        *args,
        **kwargs,
    ):
        ...


class ChatModelInterface(ABC):

    @abstractmethod
    def respond():
        ...

    @abstractmethod
    async def arespond():
        ...

    @abstractmethod
    def tokenise():
        ...

    @abstractmethod
    def tokenise_multiple():
        ...


class EmbeddingModelInterface(ABC):

    @abstractmethod
    def transform():
        ...

    @abstractmethod
    async def atransform():
        ...

    @abstractmethod
    def transform_multiple():
        ...

    @abstractmethod
    async def atransform_multiple():
        ...