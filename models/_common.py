from typing import Mapping

from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

DEFAULT_HTTP_BASE_ADDRESS = "https://aihub.useasy.cn/v1"
DEFAULT_WS_BASE_ADDRESS = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


def get_http_base_address(credentials: Mapping[str, str]) -> str:
    base_url = credentials.get("base_url")
    if base_url:
        return base_url
    return DEFAULT_HTTP_BASE_ADDRESS


def get_ws_base_address(credentials: Mapping[str, str]) -> str:
    base_url = credentials.get("base_url")
    if base_url:
        return base_url
    return DEFAULT_WS_BASE_ADDRESS


class _CommonUseasy:
    @staticmethod
    def _to_credential_kwargs(credentials: dict) -> dict:
        credentials_kwargs = {
            "useasy_api_key": credentials["useasy_api_key"],
        }

        return credentials_kwargs

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {
            InvokeConnectionError: [
                RequestFailure,
            ],
            InvokeServerUnavailableError: [
                ServiceUnavailableError,
            ],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [
                AuthenticationError,
            ],
            InvokeBadRequestError: [
                InvalidParameter,
                UnsupportedModel,
                UnsupportedHTTPMethod,
            ],
        }
