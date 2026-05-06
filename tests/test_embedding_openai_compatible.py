from types import SimpleNamespace
from unittest.mock import Mock, patch

from models.text_embedding.text_embedding import UseasyTextEmbeddingModel


def test_embedding_uses_openai_compatible_endpoint() -> None:
    embeddings = Mock()
    embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(index=1, embedding=[0.3, 0.4]),
            SimpleNamespace(index=0, embedding=[0.1, 0.2]),
        ],
        usage=SimpleNamespace(total_tokens=7),
    )
    client = Mock()
    client.embeddings = embeddings

    with patch(
        "models.text_embedding.text_embedding.OpenAI", return_value=client
    ) as openai:
        result, tokens = UseasyTextEmbeddingModel.embed_documents(
            credentials_kwargs={"useasy_api_key": "test-key"},
            model="text-embedding-v4",
            texts=["first", "second"],
            base_address="https://new-api.example.com/v1",
        )

    openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://new-api.example.com/v1",
    )
    embeddings.create.assert_called_once()
    assert embeddings.create.call_args.kwargs["model"] == "text-embedding-v4"
    assert embeddings.create.call_args.kwargs["input"] == ["first", "second"]
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert tokens == 7
