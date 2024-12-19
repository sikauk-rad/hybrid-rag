from beartype import beartype
from .datatypes import OpenAIMessageCountType
from operator import itemgetter
from numpy import fromiter

def check_all_arguments_are_none_or_not(
    *args,
) -> bool:

    """
    Check if all provided arguments are either None or not None.

    Args:
        *args: A variable number of arguments to check.

    Returns:
        bool: True if all arguments are None or all are not None; False otherwise.
    """

    all_none = [arg is None for arg in args]
    return not (any(all_none) and (not all(all_none)))


@beartype
def get_allowed_history(
    messages: list[OpenAIMessageCountType],
    token_limit: int,
    strip_counts: bool = True,
) -> list[OpenAIMessageCountType]:


    """
    Retrieve a list of messages that fit within a specified token limit.

    Args:
        messages (list[OpenAIMessageCountType]): A list of messages, each containing a token count.
        token_limit (int): The maximum number of tokens allowed in the returned messages.
        strip_counts (bool): If True, the 'tokens' key will be removed from the returned messages (default is True).
        preserve_first_system_prompt (bool): If True, the first system prompt will be preserved in the returned messages (default is True).

    Returns:
        list[OpenAIMessageCountType]: A filtered list of messages that fit within the token limit.
    """

    if not messages:
        return messages

    token_reverse_cumsums = fromiter(
        map(itemgetter('tokens'), messages),
        dtype = 'int64',
        count = len(messages),
    )[::-1].cumsum()[::-1]

    messages = messages[(token_reverse_cumsums >= token_limit).argmin():]

    if not strip_counts:
        return messages
    else:
        return [{
            key: value for key, value in d.items() if key != 'tokens'
        } for d in messages]