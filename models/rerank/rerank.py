from typing import Optional
import dashscope
import requests
from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.rerank_model import RerankModel
from models._common import get_http_base_address
from ..constant import BURY_POINT_HEADER


class GTERerankModel(RerankModel):
    """
    Model class for GTE rerank model.
    """

    @staticmethod
    def _is_new_api_base_address(base_address: str) -> bool:
        return True

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        if model.endswith(".yaml"):
            return model.removesuffix(".yaml")
        return model

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: docs for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        model = self._normalize_model_name(model)
        if len(docs) == 0:
            return RerankResult(model=model, docs=docs)
        http_base_address = get_http_base_address(credentials)
        if self._is_new_api_base_address(http_base_address):
            return self._invoke_new_api(
                model=model,
                credentials=credentials,
                base_address=http_base_address,
                query=query,
                docs=docs,
                score_threshold=score_threshold,
                top_n=top_n,
            )
        response = dashscope.TextReRank.call(
            query=query,
            headers=BURY_POINT_HEADER,
            documents=docs,
            model=model,
            top_n=top_n,
            return_documents=True,
            api_key=credentials["useasy_api_key"],
            base_address=http_base_address,
        )
        rerank_documents = []
        if not response.output:
            return RerankResult(model=model, docs=rerank_documents)
        for _, result in enumerate(response.output.results):
            rerank_document = RerankDocument(
                index=result.index,
                score=result.relevance_score,
                text=result["document"]["text"],
            )
            if score_threshold is not None:
                if result.relevance_score >= score_threshold:
                    rerank_documents.append(rerank_document)
            else:
                rerank_documents.append(rerank_document)
        return RerankResult(model=model, docs=rerank_documents)

    def _invoke_new_api(
        self,
        model: str,
        credentials: dict,
        base_address: str,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> RerankResult:
        payload = {
            "model": model,
            "query": query,
            "documents": docs,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        try:
            response = requests.post(
                f"{base_address.rstrip('/')}/rerank",
                headers={
                    "Authorization": f"Bearer {credentials['useasy_api_key']}",
                    "Content-Type": "application/json",
                    **BURY_POINT_HEADER,
                },
                json=payload,
                timeout=300,
            )
        except requests.ConnectionError as ex:
            raise InvokeConnectionError(str(ex)) from ex
        except requests.Timeout as ex:
            raise InvokeServerUnavailableError(str(ex)) from ex
        except requests.RequestException as ex:
            raise InvokeConnectionError(str(ex)) from ex

        if response.status_code >= 400:
            self._handle_new_api_error(response, model)

        try:
            data = response.json()
        except ValueError as ex:
            raise InvokeBadRequestError(
                f"Failed to parse rerank model {model} response: {response.text}"
            ) from ex
        raw_results = data.get("results")
        if raw_results is None:
            raw_results = data.get("data", [])

        rerank_documents = []
        for result in raw_results or []:
            index = result.get("index")
            if index is None:
                index = result.get("document_index")
            if index is None:
                continue

            score = result.get("relevance_score")
            if score is None:
                score = result.get("score")
            if score is None:
                score = result.get("relevanceScore", 0)

            text = ""
            document = result.get("document")
            if isinstance(document, dict):
                text = document.get("text") or document.get("content") or ""
            elif isinstance(document, str):
                text = document
            if not text and 0 <= index < len(docs):
                text = docs[index]

            score = float(score)
            if score_threshold is not None and score < score_threshold:
                continue
            rerank_documents.append(
                RerankDocument(index=int(index), score=score, text=text)
            )

        return RerankResult(model=model, docs=rerank_documents)

    def _handle_new_api_error(self, response: requests.Response, model: str) -> None:
        try:
            error = response.json()
        except ValueError:
            error = response.text

        message = error
        if isinstance(error, dict):
            message = error.get("message") or error.get("error") or error
            if isinstance(message, dict):
                message = message.get("message") or message

        error_msg = (
            f"Failed to invoke rerank model {model}, "
            f"status code: {response.status_code}, message: {message}"
        )
        if response.status_code in (401, 403):
            raise InvokeAuthorizationError(error_msg)
        if response.status_code == 429:
            raise InvokeRateLimitError(error_msg)
        if response.status_code >= 500:
            raise InvokeServerUnavailableError(error_msg)
        raise InvokeBadRequestError(error_msg)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            self.invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
                    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
                ],
                score_threshold=0.8,
            )
        except Exception as ex:
            print(ex)
            raise CredentialsValidateFailedError(str(ex))

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
            InvokeConnectionError: [RequestFailure],
            InvokeServerUnavailableError: [ServiceUnavailableError],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [AuthenticationError],
            InvokeBadRequestError: [InvalidParameter, UnsupportedModel, UnsupportedHTTPMethod],
        }
