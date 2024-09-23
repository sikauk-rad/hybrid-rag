import polars as pl
from polars.datatypes.classes import DataTypeClass
import numpy as np
from typing import Any
from beartype import beartype
from collections.abc import Iterable
from asyncio import gather
from numbers import Number
from numpy.typing import NDArray
from polars.exceptions import ColumnNotFoundError
from sentence_transformers import CrossEncoder
from .base import TextTransformer
from .exceptions import SizeError


@beartype
class DocumentScorer:

    def __init__(
        self,
        documents: list[str],
        document_sizes: list[int] | NDArray[np.integer],
        transformers: dict[str, TextTransformer],
        *,
        similarity_weights: dict[str, Number] = {},
        transform_arguments: dict[str, dict[str, Any]] = {},
        score_arguments: dict[str, dict[str, Any]] = {},
        metadata: dict[str, list[str | Number]] = {},
        reranker_name: str = 'ms-marco-MiniLM-L-6-v2',
    ) -> None:

        if len(documents) != len(document_sizes):
            raise SizeError(
                f'number of documents {len(documents)} does not match number of '\
                f'document sizes {len(document_sizes)}.'
            )

        self.document_table = pl.DataFrame(
            {
                'content': documents,
                'content size': document_sizes,
            } | metadata,
        ).with_row_index(
            name = '__index__',
        )

        self.transform_arguments, self.score_arguments, self.similarity_weights = {},{},{}
        for column_name in transformers.keys():
            self.transform_arguments[column_name] = transform_arguments.get(
                column_name, 
                {},
            )
            self.score_arguments[column_name] = score_arguments.get(
                column_name,
                {}
            )
            self.similarity_weights[f'{column_name} score'] = similarity_weights.get(
                column_name,
                1.,
            )
        self.transformers = transformers
        self.reranker_name = reranker_name
        self.reranker = CrossEncoder(f'cross-encoder/{reranker_name}')


    # def save_to_file(
    #     self,
    #     save_path: Path,
    #     fail_on_overwrite: bool = True,
    # ) -> None:

    #     save_path.mkdir(parents = True, exist_ok = not fail_on_overwrite)
    #     save_path
        


    def check_column(
        self,
        column_name: str,
        dtype: DataTypeClass | tuple[pl.DataTypeClass, ...] | None = None,
    ) -> None:

        if column_name not in self.document_table.columns:
            raise ColumnNotFoundError(f'Column {column_name} not found in document_table.')
        if dtype and not isinstance(self.document_table[column_name].dtype, dtype):
            raise TypeError(f'Column {column_name} is not of expected datatype {dtype}.')


    def transform_query(
        self,
        query: str,
    ) -> dict[str, list[list[float]] | NDArray[np.floating]]:
        
        return {column_name: text_transformer.transform(
            query, 
            **self.transform_arguments[column_name],
        ) for column_name, text_transformer in self.transformers.items()}


    async def atransform_query(
        self,
        query: str,
    ) -> dict[str, list[list[float]] | NDArray[np.floating]]:
        
        return dict(zip(
            self.similarity_columns.keys(),
            await gather(*[
                text_transformer.atransform(
                    query,
                    **self.transform_arguments[column_name],
                ) for column_name, text_transformer in self.transformers.items()
            ]),
        ))


    def _calculate_weighted_score(
        self,
        score_table: pl.DataFrame,
        fusion_factor: Number = 1,
    ) -> pl.DataFrame:

        return score_table.with_columns(
            pl.sum_horizontal(
                ((1 / (pl.col(col).arg_sort() + fusion_factor)) * weight) for col, weight in self.similarity_weights.items()
            ).alias(
                'weighted score'
            )
        ).sort(
            by = 'weighted score',
            descending = True,
        )


    def _filter_documents(
        self,
        document_table: pl.DataFrame,
        filters: Iterable[pl.Expr],
    ) -> pl.DataFrame:

        if not filters:
            return document_table
        else:
            return document_table.filter(*filters)


    def score_documents(
        self,
        query: str,
        filters: Iterable[pl.Expr] = [],
        fusion_factor: Number = 1,
    ) -> pl.DataFrame:

        filtered_documents = self._filter_documents(self.document_table, filters)

        if not filtered_documents.shape[0]:
            return filtered_documents

        return self._calculate_weighted_score(
            filtered_documents.with_columns(
                pl.Series(
                    score_column,
                    text_transformer.score(
                        query, 
                        document_indices = filtered_documents['__index__'].to_numpy(),
                        **self.transform_arguments[column_name],
                    ),
                    dtype = pl.Float32,
                ) for (column_name, text_transformer), score_column in zip(
                    self.transformers.items(),
                    self.similarity_weights.keys(),
                )
            ),
            fusion_factor = fusion_factor,
        )


    async def ascore_documents(
        self,
        query: str,
        filters: Iterable[pl.Expr] = [],
        fusion_factor: Number = 1,
    ) -> pl.DataFrame:

        filtered_documents = self._filter_documents(self.document_table, filters)

        if not filtered_documents.shape[0]:
            return filtered_documents

        document_scores = await gather(*[text_transformer.score(
            query,
            document_indices = filtered_documents['__index__'].to_numpy(),
            **self.transform_arguments[column_name],
        ) for column_name, text_transformer in self.transformers.items()])
        return self._calculate_weighted_score(
            filtered_documents.with_columns(
                pl.Series(
                    score_column,
                    document_score,
                    dtype = pl.Float32,
                ) for score_column, document_score in zip(
                    self.similarity_weights.keys(),
                    document_scores,
                )
            ),
            fusion_factor = fusion_factor,
        )


    def rank_documents(
        self,
        query: str,
        documents: Iterable[str],
    ) -> NDArray[np.floating]:

        return self.reranker.predict(
            [(query, document) for document in documents]
        )


    def rank_sort_filter_documents(
        self,
        query: str,
        document_frame: pl.DataFrame,
        rerank_score_threshold: Number,
    ) -> pl.DataFrame:

        return document_frame.with_columns(
            pl.Series(
                'rerank score',
                self.rank_documents(query, document_frame['content'].to_list()),
                dtype = pl.Float32,
            ),
        ).filter(
            pl.col('rerank score') >= rerank_score_threshold
        ).sort(
            by = 'rerank score'
        )


    def get_top_k_documents(
        self,
        query: str,
        k: int,
        weighted_score_threshold: Number,
        content_size_limit: Number,
        filters: Iterable[pl.Expr] = [],
        fusion_factor: Number = 1,
        rerank: bool = True,
        rerank_score_threshold: Number = -1.,
    ) -> pl.DataFrame:

        relevant_documents = self.score_documents(
            query, 
            filters, 
            fusion_factor = fusion_factor,
        )
        filtered_documents = relevant_documents.filter(
            pl.col('weighted score') > weighted_score_threshold
        )[:k]
        if rerank and filtered_documents.shape[0]:
            filtered_documents = self.rank_sort_filter_documents(
                query,
                filtered_documents,
                rerank_score_threshold,
            )
        return filtered_documents.filter(
            pl.col('content size').cum_sum() < content_size_limit,
        )


    async def aget_top_k_documents(
        self,
        query: str,
        k: int,
        weighted_score_threshold: Number,
        content_size_limit: Number,
        filters: Iterable[pl.Expr] = [],
        fusion_factor: Number = 1,
        rerank: bool = True,
        rerank_score_threshold: Number = -1,
    ) -> pl.DataFrame:

        relevant_documents = await self.ascore_documents(
            query, 
            filters, 
            fusion_factor = fusion_factor,
        )
        filtered_documents = relevant_documents.filter(
            pl.col('weighted score') > weighted_score_threshold
        )[:k]
        if rerank and filtered_documents.shape[0]:
            filtered_documents = self.rank_sort_filter_documents(
                query,
                filtered_documents,
                rerank_score_threshold,
            )
        return filtered_documents.filter(
            pl.col('content size').cum_sum() < content_size_limit,
        )