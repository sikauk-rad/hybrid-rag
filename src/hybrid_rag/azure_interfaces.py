from pathlib import Path
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from openai._exceptions import RateLimitError, BadRequestError
from beartype import beartype
from tqdm.asyncio import tqdm_asyncio
from asyncio import sleep as asleep
from time import sleep
from numbers import Number
from .base import EmbeddingModelInterface, ChatModelInterface, TokeniserInterface
from .datatypes import OpenAIMessageCountType, OpenAIMessageType
from .utilities import get_allowed_history
from .text_transformers.embedding_transformers import EmbeddingCache
from datetime import date, datetime


@beartype
class AzureEmbeddingModelInterface(EmbeddingModelInterface):

    def __init__(
        self,
        sync_client: OpenAI | AzureOpenAI,
        async_client: AsyncOpenAI | AsyncAzureOpenAI,
        model_name: str,
        cache: EmbeddingCache,
        base_model_name: str | None = None,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.cache = cache
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.bad_requests = []
        self.function = 'embedding'


    def update_cache(
        self,
        new_items: dict[str, list[float]],
    ) -> None:

        self.cache.update(new_items)


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
                    model = self.model_name,
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
                    model = self.model_name,
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
                desc = f'embedding with {self.base_model_name}'
            )

        finally:
            if save_path:
                self.cache.save(save_path, fail_on_overwrite)

        return embeddings


@beartype
class AzureChatModelInterface(ChatModelInterface):

    def __init__(
        self,
        sync_client: OpenAI | AzureOpenAI,
        async_client: AsyncOpenAI | AsyncAzureOpenAI,
        model_name: str,
        tokeniser: TokeniserInterface,
        token_input_limit: int,
        token_output_limit: int | None = None,
        base_model_name: str | None = None,
        knowledge_cutoff_date: date | datetime | None = None,
    ) -> None:

        self.sync_client = sync_client
        self.async_client = async_client
        self.model_name = model_name
        self.tokeniser = tokeniser
        self.token_input_limit = token_input_limit
        self.token_output_limit = token_output_limit
        self.base_model_name = base_model_name
        self.knowledge_cutoff_date = knowledge_cutoff_date
        self.function = 'chat'
        self.chat_parameters = {
            'model': self.model_name,
            'n': 1,
        }


    def respond(
        self,
        messages: list[OpenAIMessageType],
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
        messages: list[OpenAIMessageType],
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


    def trim_and_respond(
        self,
        messages: list[OpenAIMessageCountType],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: list[int] | None = None,
    ) -> tuple[str, int] | str:

        return self.respond(
            messages = get_allowed_history(
                messages,
                self.token_input_limit,
                message_preservation_indices = message_preservation_indices,
            ),
            temperature = temperature,
            return_token_count = return_token_count,
        )


    async def atrim_and_respond(
        self,
        messages: list[OpenAIMessageCountType],
        temperature: Number = 0,
        return_token_count: bool = False,
        message_preservation_indices: list[int] | None = None,
    ) -> tuple[str, int] | str:

        return await self.arespond(
            messages = get_allowed_history(
                messages,
                self.token_input_limit,
                message_preservation_indices = message_preservation_indices,
            ),
            temperature = temperature,
            return_token_count = return_token_count,
        )