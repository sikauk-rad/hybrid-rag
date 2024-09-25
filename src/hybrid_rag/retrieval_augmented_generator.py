import polars as pl
import numpy as np
from typing import Literal
from beartype import beartype
from collections.abc import Iterable
from numbers import Number
from .base import ChatModelInterface
from .document_scorer import DocumentScorer
from operator import itemgetter
from .types import OpenAIMessageType, OpenAIMessageCountType


@beartype
class RetrievalAugmentedGenerator:

    def __init__(
        self,
        *,
        history_chat_model: ChatModelInterface,
        question_chat_model: ChatModelInterface,
        process_chat_model: ChatModelInterface,
        document_scorer: DocumentScorer,
        history_prompts: tuple[str, str],
        question_prompts: tuple[str, str, str],
        process_prompts: tuple[str, str],
    ) -> None:

        self.history_chat_model = history_chat_model
        self.question_chat_model = question_chat_model
        self.process_chat_model = process_chat_model
        self.document_scorer = document_scorer
        self.history_prompts = history_prompts
        self.question_prompts = question_prompts
        self.process_prompts = process_prompts

        self.history_token_limit = (
            self.history_chat_model.token_input_limit 
            - 
            self.history_chat_model.get_token_length('\n'.join(self.history_prompts))
            - 
            10
        )
        self.question_token_limit = (
            self.question_chat_model.token_input_limit
            -
            self.question_chat_model.get_token_length('\n'.join(self.question_prompts))
            -
            10
        )
        self.chat_history = []
        self.interaction_history = []
        self.token_getter = itemgetter('tokens')


    def _get_allowed_history(
        self,
        token_limit: int,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> list[OpenAIMessageType]:

        if custom_history is None:
            history = self.chat_history
        else:
            history = custom_history
        if (not history) or (token_limit < 2):
            return []

        token_counts = np.cumsum(
            [*map(self.token_getter, history)],
        )
        token_count_index = (token_counts > token_limit).argmax()
        return [{
            'role': d['role'], 
            'content': d['content'],
        } for d in history[-token_count_index:]]


    def make_standalone_question(
        self,
        query: str,
        temperature: Number,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> tuple[str, int]:

        query_length = self.history_chat_model.get_token_length(query)
        messages = [
            {'role': 'system', 'content': self.history_prompts[0]},
            *self._get_allowed_history(
                self.history_token_limit - query_length,
                custom_history = custom_history,
            ),
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': self.history_prompts[1]},
        ]
        response, token_count = self.history_chat_model.respond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, query_length)


    async def amake_standalone_question(
        self,
        query: str,
        temperature: Number,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> tuple[str, int]:

        query_length = self.history_chat_model.get_token_length(query)
        messages = [
            {'role': 'system', 'content': self.history_prompts[0]},
            *self._get_allowed_history(
                self.history_token_limit - query_length,
                custom_history = custom_history,
            ),
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': self.history_prompts[1]},
        ]
        response, token_count = await self.history_chat_model.arespond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, query_length)


    def respond_to_standalone_question(
        self,
        query: str,
        temperature: Number,
        history_token_limit: int,
        n_documents: int,
        initial_retrieval_ratio: Number,
        weighted_rank_threshold: float,
        fusion_factor: Number,
        rerank: bool,
        rerank_score_threshold: Number,
        filters: Iterable[pl.Expr] = [],
        verbose: bool = False,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> tuple[str, int]:

        query_length = self.question_chat_model.get_token_length(query)
        context_token_limit = self.question_token_limit - query_length - history_token_limit
        relevant_documents = self.document_scorer.get_top_k_documents(
            query = query,
            k = n_documents,
            k_multiplier = initial_retrieval_ratio,
            fusion_factor = fusion_factor,
            weighted_rank_threshold = weighted_rank_threshold,
            filters = filters,
            rerank = rerank,
            rerank_score_threshold = rerank_score_threshold,
            content_size_limit = context_token_limit,
            verbose = verbose,
        ).get_column('content').to_list()
        messages = [
            {'role': 'system', 'content': self.question_prompts[0]},
            *self._get_allowed_history(
                token_limit = history_token_limit,
                custom_history = custom_history,
            ),
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': '\n\n'.join([
                self.question_prompts[1],
                *[f'DOCUMENT {n}:\n{text}' for n, text in enumerate(relevant_documents, 1)],
                self.question_prompts[2],
            ])},
        ]
        response, token_count = self.question_chat_model.respond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, token_count)


    async def arespond_to_standalone_question(
        self,
        query: str,
        temperature: Number,
        history_token_limit: int,
        n_documents: int,
        initial_retrieval_ratio: Number,
        weighted_rank_threshold: float,
        fusion_factor: Number,
        rerank: bool,
        rerank_score_threshold: Number,
        filters: Iterable[pl.Expr] = [],
        verbose: bool = False,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> tuple[str, int]:

        query_length = self.question_chat_model.get_token_length(query)
        context_token_limit = self.question_token_limit - query_length - history_token_limit
        relevant_documents = (await self.document_scorer.aget_top_k_documents(
            query = query,
            k = n_documents,
            k_multiplier = initial_retrieval_ratio,
            fusion_factor = fusion_factor,
            weighted_rank_threshold = weighted_rank_threshold,
            filters = filters,
            rerank = rerank,
            rerank_score_threshold = rerank_score_threshold,
            content_size_limit = context_token_limit,
            verbose = verbose,
        )).get_column('content').to_list()
        messages = [
            {'role': 'system', 'content': self.question_prompts[0]},
            *self._get_allowed_history(
                token_limit = history_token_limit,
                custom_history = custom_history,
            ),
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': '\n\n'.join([
                self.question_prompts[1],
                *[f'DOCUMENT {n}:\n{text}' for n, text in enumerate(relevant_documents, 1)],
                self.question_prompts[2],
            ])},
        ]
        response, token_count = await self.question_chat_model.arespond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, token_count)


    def process_response(
        self,
        query: str,
        target_language: str,
        temperature: Number,
        history_token_limit: int,
        custom_history: list[OpenAIMessageCountType] | None = None,
    ) -> tuple[str, int]:

        messages = [
            *self._get_allowed_history(
                token_limit = history_token_limit,
                custom_history = custom_history,
            ),
            {
                'role': 'system', 
                'content': '\n'.join([
                    self.process_prompts[0].replace('<<target_language>>', target_language),
                    query,
                    self.process_prompts[1].replace('<<target_language>>', target_language),
                ]),
            },
        ]
        response, token_count = self.process_chat_model.respond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, token_count)


    async def aprocess_response(
        self,
        query: str,
        target_language: str,
        temperature: Number,
        history_token_limit: int,
        custom_history: list[OpenAIMessageType] | None = None,
    ) -> tuple[str, int]:

        messages = [
            *self._get_allowed_history(
                token_limit = history_token_limit,
                custom_history = custom_history,
            ),
            {
                'role': 'system', 
                'content': '\n'.join([
                    self.process_prompts[0].replace('<<target_language>>', target_language),
                    query,
                    self.process_prompts[1].replace('<<target_language>>', target_language),
                ]),
            },
        ]
        response, token_count = await self.process_chat_model.arespond(
            messages = messages,
            temperature = temperature,
            return_token_count = True,
        )
        self.interaction_history.append(([*messages, {
            'role': 'assistant',
            'content': response,
        }]))
        return (response, token_count)


    def respond_to_query(
        self,
        query: str,
        history_model_temperature: Number = 0,
        history_model_custom_history:  list[OpenAIMessageType] | None = None,
        question_model_temperature: Number = 0,
        question_model_history_token_limit: int = 1_000,
        question_model_custom_history:  list[OpenAIMessageType] | None = None,
        n_documents: int = 50,
        initial_retrieval_ratio: Number = 2., 
        fusion_factor: Number = 1,
        weighted_rank_threshold: float = 0.001,
        filters: Iterable[pl.Expr] = [],
        rerank: bool = True,
        rerank_score_threshold: Number = -np.inf,
        process_model_target_language: str = 'British English',
        process_model_temperature: Number = 0,
        process_model_history_token_limit: int = 0,
        process_model_custom_history:  list[OpenAIMessageType] | None = None,
        return_token_count: bool = False,
        verbose: bool = False,
    ) -> str | tuple[str, int]:

        standalone_query, query_length = self.make_standalone_question(
            query,
            temperature = history_model_temperature,
            custom_history = history_model_custom_history,
        )
        if verbose:
            print(f'initial query: {query}')
            print(f'standalone question: {standalone_query}')
        response, token_count = self.respond_to_standalone_question(
            standalone_query,
            temperature = question_model_temperature,
            n_documents = n_documents,
            initial_retrieval_ratio = initial_retrieval_ratio,
            fusion_factor = fusion_factor,
            weighted_rank_threshold = weighted_rank_threshold,
            filters = filters,
            rerank = rerank,
            rerank_score_threshold = rerank_score_threshold,
            history_token_limit = question_model_history_token_limit,
            custom_history = question_model_custom_history,
            verbose = verbose,
        )
        if verbose:
            print(f'initial response: {response}')
        processed_response, token_count = self.process_response(
            response,
            target_language = process_model_target_language,
            temperature = process_model_temperature,
            history_token_limit = process_model_history_token_limit,
            custom_history = process_model_custom_history,
        )
        self.chat_history.append({
            'role': 'user', 
            'content': query,
            'tokens': query_length,
        })
        self.chat_history.append({
            'role': 'assistant', 
            'content': processed_response,
            'tokens': token_count,
        })
        return (processed_response, token_count) if return_token_count else processed_response


    async def arespond_to_query(
        self,
        query: str,
        history_model_temperature: Number = 0,
        history_model_custom_history:  list[OpenAIMessageType] | None = None,
        question_model_temperature: Number = 0,
        question_model_history_token_limit: int = 1_000,
        question_model_custom_history:  list[OpenAIMessageType] | None = None,
        n_documents: int = 50,
        initial_retrieval_ratio: Number = 2., 
        fusion_factor: Number = 1,
        weighted_rank_threshold: float = 0.001,
        filters: Iterable[pl.Expr] = [],
        rerank: bool = True,
        rerank_score_threshold: Number = -np.inf,
        process_model_target_language: str = 'British English',
        process_model_temperature: Number = 0,
        process_model_history_token_limit: int = 0,
        process_model_custom_history:  list[OpenAIMessageType] | None = None,
        return_token_count: bool = False,
        verbose: bool = False,
    ) -> str | tuple[str, int]:

        standalone_query, query_length = await self.amake_standalone_question(
            query,
            temperature = history_model_temperature,
            custom_history = history_model_custom_history,
        )
        if verbose:
            print(f'initial query: {query}')
            print(f'standalone question: {standalone_query}')
        response, token_count = await self.arespond_to_standalone_question(
            standalone_query,
            temperature = question_model_temperature,
            n_documents = n_documents,
            initial_retrieval_ratio = initial_retrieval_ratio,
            fusion_factor = fusion_factor,
            weighted_rank_threshold = weighted_rank_threshold,
            filters = filters,
            rerank = rerank,
            rerank_score_threshold = rerank_score_threshold,
            history_token_limit = question_model_history_token_limit,
            custom_history = question_model_custom_history,
            verbose = verbose,
        )
        if verbose:
            print(f'initial response: {response}')
        processed_response, token_count = await self.aprocess_response(
            response,
            target_language = process_model_target_language,
            temperature = process_model_temperature,
            history_token_limit = process_model_history_token_limit,
            custom_history = process_model_custom_history,
        )
        self.chat_history.append({
            'role': 'user', 
            'content': query,
            'tokens': query_length,
        })
        self.chat_history.append({
            'role': 'assistant', 
            'content': processed_response,
            'tokens': token_count,
        })
        return (processed_response, token_count) if return_token_count else processed_response