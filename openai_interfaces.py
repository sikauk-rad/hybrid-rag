from pathlib import Path
import numpy as np
import openai
from openai._exceptions import RateLimitError, BadRequestError
import tiktoken
from typing import Literal
from beartype import beartype
from tqdm.asyncio import tqdm_asyncio
from asyncio import sleep as asleep
from time import sleep
from numbers import Number
from numpy.typing import NDArray
from .base import EmbeddingModelInterface, ChatModelInterface
from .text_transformers.embedding_transformers import EmbeddingCache


@beartype
class OpenAIEmbeddingModelInterface(EmbeddingModelInterface):

    def __init__(
        self,
        sync_client: openai.OpenAI | openai.AzureOpenAI,
        async_client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI,
        model: str,
        cache: EmbeddingCache,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.cache = cache
        self.model = model
        self.bad_requests = []


    def transform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding
        
        for _ in range(n_retries):
            try:
                embedding = self.sync_client.embeddings.create(
                    model = self.model,
                    input = text,
                ).data[0].embedding
                self.cache.add(text, embedding)
                break
            except RateLimitError:
                sleep(1)
            except BadRequestError:
                self.bad_requests.append(text)
                break
        else:
            return []

        return embedding
    

    async def atransform(
        self,
        text: str,
        n_retries: int = 100,
    ) -> list[float]:

        embedding = self.cache.retrieve(text)
        if embedding:
            return embedding

        for _ in range(n_retries):
            try:
                embedding = (await self.async_client.embeddings.create(
                    model = self.model,
                    input = text,
                )).data[0].embedding
                self.cache.add(text, embedding)
                break
            except RateLimitError:
                await asleep(1)
            except BadRequestError:
                self.bad_requests.append(text)
                break

        else:
            return []

        return embedding


    def transform_multiple(
        self,
        texts: list[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:
    
        try:
            embeddings = [self.transform(
                text = text,
                n_retries = n_retries,
            ) for text in texts]

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


    async def atransform_multiple(
        self,
        texts: list[str],
        n_retries: int = 100,
        save_path: Path | None = None,
        fail_on_overwrite: bool = True,
    ) -> list[list[float]]:


        try:
            coroutines = [self.atransform(
                text = text,
                n_retries = n_retries,
            ) for text in texts]
            embeddings = await tqdm_asyncio.gather(
                *coroutines,
                position = 0,
                leave = True,
                desc = f'embedding with {self.model}'
            )

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


@beartype
class OpenAIChatModelInterface(ChatModelInterface):

    def __init__(
        self,
        sync_client: openai.OpenAI | openai.AzureOpenAI,
        async_client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI,
        model_name: str,
        tokeniser: tiktoken.core.Encoding,
        token_input_limit: int,
        token_output_limit: int | None = None,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.model_name = model_name
        self.tokeniser = tokeniser
        self.token_input_limit = token_input_limit
        self.token_output_limit = token_output_limit
        self.chat_parameters = {
            'model': self.model_name,
            'n': 1,
        }


    def respond(
        self,
        messages: list[dict[Literal['role', 'content'], str]],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[str, int] | str:

        response = self.sync_client.chat.completions.create(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
        )
        answer = response.choices[0].message.content
        return (answer, response.usage.completion_tokens) if return_token_count else answer


    async def arespond(
        self,
        messages: list[dict[Literal['role', 'content'], str]],
        temperature: Number = 0,
        return_token_count: bool = False,
    ) -> tuple[str, int] | str:

        response = await self.async_client.chat.completions.create(
            **self.chat_parameters,
            temperature = temperature,
            messages = messages,
        )
        answer = response.choices[0].message.content
        return (answer, response.usage.completion_tokens) if return_token_count else answer


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


    def get_token_length_multiple(
        self,
        texts: list[str],
    ) -> list[int]:

        return [*map(len, self.tokenise_multiple(texts))]