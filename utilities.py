from beartype import beartype
import openai

def check_all_arguments_are_none_or_not(
    *args,
) -> bool:

    all_none = [arg is None for arg in args]
    return not (any(all_none) and (not all(all_none)))


@beartype
def load_openai_clients(
    api_key: str,
    use_azure: bool,
    api_version: str | None = None,
    azure_endpoint: str | None = None,
    max_retries: int = 1000,
) -> tuple[openai.OpenAI | openai.AzureOpenAI, openai.AsyncOpenAI | openai.AsyncAzureOpenAI]:

    """
    Description
    -----------
    Load OpenAI or Azure OpenAI clients for synchronous and asynchronous operations.
    
    This function initializes and returns a tuple containing both synchronous and asynchronous clients for interacting with the OpenAI API or Azure OpenAI API, based on the provided parameters.

    Parameters
    ----------
    - `api_key` (str):  
    The API key required to authenticate with the OpenAI or Azure OpenAI service.

    - `use_azure` (bool):  
    A flag indicating whether to use Azure OpenAI (`True`) or standard OpenAI (`False`).  
    - If `True`, Azure-specific parameters (`api_version` and `azure_endpoint`) must be provided.

    - `api_version` (str, optional):  
    The version of the Azure OpenAI API to use.  
    - **Required** if `use_azure` is `True`.  
    - **Default**: `None`.

    - `azure_endpoint` (str, optional):  
    The endpoint URL for the Azure OpenAI service.  
    - **Required** if `use_azure` is `True`.  
    - **Default**: `None`.

    - `max_retries` (int, optional):  
    The maximum number of retries for API requests in case of transient failures.  
    - **Default**: `1000`.

    Returns
    -------
    - `tuple[openai.OpenAI | openai.AzureOpenAI, openai.AsyncOpenAI | openai.AsyncAzureOpenAI]`:  
    A tuple containing two clients:
    - The first element is a synchronous client (`openai.OpenAI` or `openai.AzureOpenAI`).
    - The second element is an asynchronous client (`openai.AsyncOpenAI` or `openai.AsyncAzureOpenAI`).

    Raises
    ------
    - `TypeError`:  
    If `use_azure` is `True` and either `api_version` or `azure_endpoint` is not provided.

    Example Usage:
    --------------
    # Example of loading standard OpenAI clients
    sync_client, async_client = load_azure_openai_clients(
        api_key='your-openai-api-key',
        use_azure=False,
    )
    # Example of loading Azure OpenAI clients
    sync_azure_client, async_azure_client = load_azure_openai_clients(
        api_key='your-azure-openai-api-key',
        use_azure=True,
        api_version='2023-05-15',
        azure_endpoint='https://your-azure-endpoint',
    )
    """

    if not use_azure:
        return (
            openai.OpenAI(
                api_key = api_key,
                max_retries = max_retries,
            ),
            openai.AsyncOpenAI(
                api_key = api_key,
                max_retries = max_retries,
            )
        )
    
    if not api_version or not azure_endpoint:
        raise TypeError(
            'for AzureOpenAI, azure_version and azure_endpoint must be provided.'
        )

    return (
        openai.AzureOpenAI(
            api_key = api_key,
            azure_endpoint = azure_endpoint,
            api_version = api_version,
            max_retries = max_retries,
        ),
        openai.AsyncAzureOpenAI(
            api_key = api_key,
            azure_endpoint = azure_endpoint,
            api_version = api_version,
            max_retries = max_retries,
        ),
    )