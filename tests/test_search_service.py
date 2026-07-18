import unittest

from search_service import (
    OfficialSourceRegistry,
    SearchResponseError,
    SearchService,
)


class FakeResponse:
    def __init__(self, payload, *, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, *, post_payload=None, get_payload=None, post_error=None):
        self.post_payload = post_payload
        self.get_payload = get_payload
        self.post_error = post_error
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return FakeResponse(self.post_payload)

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return FakeResponse(self.get_payload)


class SearchServiceTests(unittest.TestCase):
    def test_official_search_uses_domain_allowlist_and_exact_social_filter(self):
        http = FakeHTTPClient(
            post_payload={
                "results": [
                    {
                        "title": "BM-ECHOES official update",
                        "url": "https://bm-echoes.com/news/123#fragment",
                        "content": "Official-page snippet that is still untrusted input.",
                        "published_date": "2026-07-16",
                        "score": 0.91,
                    },
                    {
                        "title": "Exact personal account",
                        "url": "https://x.com/aoki__hina/status/123",
                        "content": "An exact allowlisted account path.",
                    },
                    {
                        "title": "Wrong X account",
                        "url": "https://x.com/not_hina/status/123",
                        "content": "Must not pass the exact-account filter.",
                    },
                    {
                        "title": "Wrong HiBiKi artist",
                        "url": "https://hibiki-cast.jp/hibiki_f/another_person/",
                        "content": "An unrelated artist on the same official domain.",
                    },
                    {
                        "title": "Hina HiBiKi profile",
                        "url": "https://hibiki-cast.jp/hibiki_f/aoki_hina/profile",
                        "content": "A URL within the registered Hina profile path.",
                    },
                    {
                        "title": "Insecure official link",
                        "url": "http://bm-echoes.com/news/insecure",
                        "content": "HTTP must not be accepted as an official result.",
                    },
                    {
                        "title": "Fan site",
                        "url": "https://aokihina.com/news",
                        "content": "Must be blocklisted.",
                    },
                    {
                        "title": "Unsafe URL",
                        "url": "javascript:alert(1)",
                        "content": "Must be rejected by URL validation.",
                    },
                ]
            }
        )
        service = SearchService(tavily_api_key="tvly-test", http_client=http)

        response = service.search_hina_official("青木阳菜 最新活动", max_results=5)

        self.assertEqual([item.official_source for item in response.results], [
            "hina_bm_echoes",
            "hina_personal_x",
            "hina_hibiki_profile",
        ])
        self.assertFalse(response.fact_eligible)
        self.assertTrue(response.untrusted)
        self.assertTrue(all(item.untrusted for item in response.results))
        grounding = response.to_grounding_bundle()
        self.assertTrue(grounding.facts[0].untrusted)
        self.assertFalse(grounding.facts[0].fact_eligible)
        self.assertEqual(grounding.sources[0].trust_level, 100)
        self.assertTrue(grounding.sources[0].untrusted)
        self.assertNotIn("#fragment", response.results[0].url)
        payload = http.post_calls[0][1]["json"]
        self.assertIs(payload["include_answer"], False)
        self.assertIs(payload["include_raw_content"], False)
        self.assertIn("bm-echoes.com", payload["include_domains"])
        self.assertNotIn("x.com", payload["include_domains"])
        self.assertNotIn("youtube.com", payload["include_domains"])

    def test_search_web_falls_back_to_brave_but_never_becomes_fact_eligible(self):
        http = FakeHTTPClient(
            post_error=RuntimeError("simulated Tavily outage"),
            get_payload={
                "web": {
                    "results": [
                        {
                            "title": "General result",
                            "url": "https://example.com/news",
                            "description": "A general web snippet.",
                        }
                    ]
                }
            },
        )
        service = SearchService(
            tavily_api_key="tvly-test",
            brave_api_key="brave-test",
            http_client=http,
        )

        response = service.search_web("普通网页问题")

        self.assertEqual(response.provider, "brave")
        self.assertFalse(response.fact_eligible)
        self.assertEqual(len(http.post_calls), 1)
        self.assertEqual(len(http.get_calls), 1)
        self.assertEqual(
            http.get_calls[0][1]["headers"]["X-Subscription-Token"], "brave-test"
        )

    def test_malformed_provider_shape_is_rejected(self):
        service = SearchService(
            tavily_api_key="tvly-test",
            http_client=FakeHTTPClient(post_payload={"answer": "missing results"}),
        )

        with self.assertRaises(SearchResponseError):
            service.search_web("结构校验")

    def test_registry_blocks_fan_site_and_requires_social_path_boundary(self):
        registry = OfficialSourceRegistry.load()

        self.assertTrue(
            registry.is_allowed_official_url(
                "https://x.com/AokiHina_Staff/status/123"
            )
        )
        self.assertFalse(
            registry.is_allowed_official_url("https://x.com/AokiHina_StaffFake")
        )
        self.assertFalse(
            registry.is_allowed_official_url("https://youtube.com/watch?v=unrelated")
        )
        self.assertTrue(registry.is_blocked_url("https://www.aokihina.com/"))


if __name__ == "__main__":
    unittest.main()
