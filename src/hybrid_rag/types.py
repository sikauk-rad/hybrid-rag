from typing import TypedDict, Literal


class OpenAIMessageCountType(TypedDict, total = True):
    role: Literal['system', 'assistant', 'user']
    content: str
    tokens: int


class OpenAIMessageType(TypedDict, total = True):
    role: Literal['system', 'assistant', 'user']
    content: str