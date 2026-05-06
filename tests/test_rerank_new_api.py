from unittest.mock import Mock, patch

from models.rerank.rerank import GTERerankModel


def test_rerank_uses_new_api_endpoint() -> None:
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "results": [
            {
                "index": 1,
                "relevance_score": 0.9,
                "document": {"text": "second"},
            },
            {
                "index": 0,
                "relevance_score": 0.2,
                "document": {"text": "first"},
            },
        ]
    }

    with patch("models.rerank.rerank.requests.post", return_value=response) as post:
        result = GTERerankModel(model_schemas=[])._invoke(
            model="gte-rerank",
            credentials={
                "useasy_api_key": "test-key",
                "base_url": "https://dev1-aihub.useasy.cn/v1",
            },
            query="query",
            docs=["first", "second"],
            score_threshold=0.5,
            top_n=1,
        )

    post.assert_called_once()
    assert post.call_args.args[0] == "https://dev1-aihub.useasy.cn/v1/rerank"
    assert post.call_args.kwargs["json"] == {
        "model": "gte-rerank",
        "query": "query",
        "documents": ["first", "second"],
        "top_n": 1,
    }
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert len(result.docs) == 1
    assert result.docs[0].index == 1
    assert result.docs[0].score == 0.9
    assert result.docs[0].text == "second"
