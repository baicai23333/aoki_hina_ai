"""Bounded external search with an explicit official-source registry.

Search snippets are always untrusted input.  Strictly allowlisted official
results may support factual grounding, but neither official nor general web
content is ever treated as an instruction.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import requests

from grounding import GroundingBundle


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("official_sources.json")
TAVILY_ENDPOINT = "https://api.tavily.com/search"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SOCIAL_PLATFORM_DOMAINS = frozenset({"x.com", "youtube.com"})

MAX_QUERY_CHARS = 300
MAX_RESULTS = 8
MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 2_000
MAX_URL_CHARS = 2_048
DEFAULT_TIMEOUT_SECONDS = 12.0

_DOMAIN_PATTERN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))+$"
)
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class SearchServiceError(RuntimeError):
    """Base class for safe search failures."""


class SearchConfigurationError(SearchServiceError):
    """Raised when the selected provider is not configured."""


class SearchValidationError(SearchServiceError, ValueError):
    """Raised when a caller supplies an invalid search request."""


class SearchResponseError(SearchServiceError):
    """Raised when a provider response violates the expected structure."""


@dataclass(frozen=True)
class OfficialSource:
    name: str
    source_type: str
    base_url: str
    domain: str
    trust_level: int
    categories: tuple[str, ...]
    exact_path: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """One sanitized web result.  ``untrusted`` is intentionally immutable."""

    title: str
    url: str
    snippet: str
    published_at: str | None = None
    score: float | None = None
    official_source: str | None = None
    trust_level: int | None = None
    untrusted: bool = True


@dataclass(frozen=True)
class SearchResponse:
    query: str
    provider: str
    results: tuple[SearchResult, ...]
    fact_eligible: bool = False
    untrusted: bool = True

    def to_grounding_bundle(self) -> GroundingBundle:
        return GroundingBundle.from_search_response(self)


def _required_string(value: object, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise SearchValidationError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise SearchValidationError(f"{field_name} cannot be empty")
    if len(cleaned) > max_length:
        raise SearchValidationError(
            f"{field_name} cannot exceed {max_length} characters"
        )
    return cleaned


def _normalize_domain(value: object) -> str:
    domain = _required_string(value, "domain", 253).lower().rstrip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    if not _DOMAIN_PATTERN.fullmatch(domain):
        raise SearchValidationError(f"invalid domain: {domain!r}")
    return domain


def _bounded_result_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SearchValidationError("max_results must be an integer")
    if not 1 <= value <= MAX_RESULTS:
        raise SearchValidationError(
            f"max_results must be between 1 and {MAX_RESULTS}"
        )
    return value


def _hostname_for_url(url: str) -> str | None:
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return None
    if hostname is None:
        return None
    normalized = hostname.lower().rstrip(".")
    return normalized[4:] if normalized.startswith("www.") else normalized


def _domain_matches(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith(f".{domain}")


class OfficialSourceRegistry:
    """Validated official sources and a hard denylist."""

    def __init__(
        self,
        sources: Iterable[OfficialSource],
        *,
        blocked_domains: Iterable[str] = (),
    ) -> None:
        normalized_sources = tuple(sources)
        if not normalized_sources:
            raise SearchValidationError("official source registry cannot be empty")
        names = [source.name for source in normalized_sources]
        if len(names) != len(set(names)):
            raise SearchValidationError("official source names must be unique")
        self.sources = normalized_sources
        self.blocked_domains = frozenset(
            _normalize_domain(domain) for domain in blocked_domains
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTRY_PATH) -> "OfficialSourceRegistry":
        registry_path = Path(path)
        try:
            raw = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchConfigurationError(
                f"cannot load official source registry: {registry_path}"
            ) from exc
        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise SearchConfigurationError("official source registry version must be 1")
        raw_sources = raw.get("sources")
        raw_blocklist = raw.get("blocklist", [])
        if not isinstance(raw_sources, list) or not isinstance(raw_blocklist, list):
            raise SearchConfigurationError(
                "official source registry must contain sources and blocklist arrays"
            )

        sources: list[OfficialSource] = []
        for index, item in enumerate(raw_sources):
            if not isinstance(item, dict):
                raise SearchConfigurationError(f"source {index} must be an object")
            name = _required_string(item.get("name"), f"sources[{index}].name", 80)
            source_type = _required_string(
                item.get("source_type"), f"sources[{index}].source_type", 32
            )
            if source_type not in {"web", "social_exact"}:
                raise SearchConfigurationError(
                    f"unsupported source_type for {name}: {source_type}"
                )
            base_url = _required_string(
                item.get("base_url"), f"sources[{index}].base_url", MAX_URL_CHARS
            )
            parsed_base = urlsplit(base_url)
            if parsed_base.scheme != "https" or not parsed_base.hostname:
                raise SearchConfigurationError(
                    f"official source {name} must use an https base_url"
                )
            domain = _normalize_domain(item.get("domain"))
            base_hostname = _hostname_for_url(base_url)
            if base_hostname is None or not _domain_matches(base_hostname, domain):
                raise SearchConfigurationError(
                    f"base_url host does not match source domain for {name}"
                )
            trust_level = item.get("trust_level")
            if (
                isinstance(trust_level, bool)
                or not isinstance(trust_level, int)
                or not 0 <= trust_level <= 100
            ):
                raise SearchConfigurationError(
                    f"trust_level for {name} must be an integer from 0 to 100"
                )
            raw_categories = item.get("categories")
            if not isinstance(raw_categories, list) or not raw_categories:
                raise SearchConfigurationError(f"categories for {name} must be a list")
            categories = tuple(
                _required_string(value, f"{name}.category", 80)
                for value in raw_categories
            )
            exact_path = item.get("exact_path")
            if source_type == "social_exact":
                if domain not in SOCIAL_PLATFORM_DOMAINS:
                    raise SearchConfigurationError(
                        f"social_exact source {name} uses an unsupported platform"
                    )
                if (
                    not isinstance(exact_path, str)
                    or not exact_path.startswith("/")
                    or len(exact_path) > 200
                ):
                    raise SearchConfigurationError(
                        f"social_exact source {name} requires exact_path"
                    )
            elif exact_path is not None:
                raise SearchConfigurationError(
                    f"web source {name} cannot define exact_path"
                )
            sources.append(
                OfficialSource(
                    name=name,
                    source_type=source_type,
                    base_url=base_url,
                    domain=domain,
                    trust_level=trust_level,
                    categories=categories,
                    exact_path=exact_path,
                )
            )

        blocked_domains: list[str] = []
        for index, item in enumerate(raw_blocklist):
            if not isinstance(item, dict):
                raise SearchConfigurationError(f"blocklist {index} must be an object")
            blocked_domains.append(_normalize_domain(item.get("domain")))
        return cls(sources, blocked_domains=blocked_domains)

    @property
    def ordinary_search_domains(self) -> tuple[str, ...]:
        """Domains safe to pass to a provider's domain-level allowlist."""

        return tuple(
            dict.fromkeys(
                source.domain
                for source in self.sources
                if source.source_type == "web"
                and source.domain not in SOCIAL_PLATFORM_DOMAINS
                and source.domain not in self.blocked_domains
            )
        )

    def is_blocked_url(self, url: str) -> bool:
        hostname = _hostname_for_url(url)
        return hostname is None or any(
            _domain_matches(hostname, domain) for domain in self.blocked_domains
        )

    def match_source(self, url: str) -> OfficialSource | None:
        """Match normal official domains or an exact social account path."""

        if self.is_blocked_url(url):
            return None
        try:
            parsed = urlsplit(url)
        except ValueError:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme.lower() != "https"
            or port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        hostname = _hostname_for_url(url)
        if hostname is None:
            return None
        matches: list[OfficialSource] = []
        for source in self.sources:
            if not _domain_matches(hostname, source.domain):
                continue
            if source.source_type == "web":
                base_path = urlsplit(source.base_url).path.rstrip("/") or "/"
                result_path = parsed.path.rstrip("/") or "/"
                if base_path != "/" and not (
                    result_path == base_path
                    or result_path.startswith(base_path + "/")
                ):
                    continue
                matches.append(source)
                continue
            assert source.exact_path is not None
            path = parsed.path.rstrip("/").lower()
            exact_path = source.exact_path.rstrip("/").lower()
            if path == exact_path or path.startswith(f"{exact_path}/"):
                matches.append(source)
        if not matches:
            return None
        return max(matches, key=lambda item: (item.trust_level, len(item.base_url)))

    def is_allowed_official_url(self, url: str) -> bool:
        return self.match_source(url) is not None


def _clean_text(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = _CONTROL_CHARACTERS.sub("", value)
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_length]


def _safe_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > MAX_URL_CHARS:
        return None
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        return None
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, "")
    )


class SearchService:
    """Tavily-first search with Brave limited to general-search fallback."""

    def __init__(
        self,
        *,
        tavily_api_key: str | None = None,
        brave_api_key: str | None = None,
        registry: OfficialSourceRegistry | None = None,
        http_client: Any | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.tavily_api_key = (
            tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY")
        )
        self.brave_api_key = (
            brave_api_key if brave_api_key is not None else os.getenv("BRAVE_SEARCH_API_KEY")
        )
        self.registry = registry or OfficialSourceRegistry.load()
        self.http_client = http_client or requests.Session()
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise SearchValidationError("timeout_seconds must be numeric")
        self.timeout_seconds = float(timeout_seconds)
        if not 0 < self.timeout_seconds <= 60:
            raise SearchValidationError("timeout_seconds must be between 0 and 60")

    def search_hina_official(
        self, query: str, *, max_results: int = 5
    ) -> SearchResponse:
        """Search strict official web domains; never widen to Brave or all social."""

        clean_query = _required_string(query, "query", MAX_QUERY_CHARS)
        result_limit = _bounded_result_count(max_results)
        if not self.tavily_api_key:
            raise SearchConfigurationError(
                "TAVILY_API_KEY is required for official-source search"
            )
        domains = self.registry.ordinary_search_domains
        if not domains:
            raise SearchConfigurationError("official domain allowlist is empty")
        response = self._search_tavily(
            clean_query,
            max_results=result_limit,
            include_domains=domains,
            official_only=True,
        )
        return SearchResponse(
            query=response.query,
            provider=response.provider,
            results=response.results,
            # Search-provider snippets are discovery leads, not verified page
            # facts. Only reviewed collector records may ground confirmations.
            fact_eligible=False,
        )

    def search_web(self, query: str, *, max_results: int = 5) -> SearchResponse:
        """General search.  Brave is used only when Tavily is unavailable/fails."""

        clean_query = _required_string(query, "query", MAX_QUERY_CHARS)
        result_limit = _bounded_result_count(max_results)
        tavily_error: SearchServiceError | None = None
        if self.tavily_api_key:
            try:
                return self._search_tavily(
                    clean_query,
                    max_results=result_limit,
                    include_domains=None,
                    official_only=False,
                )
            except SearchServiceError as exc:
                tavily_error = exc
        if self.brave_api_key:
            return self._search_brave(clean_query, max_results=result_limit)
        if tavily_error is not None:
            raise tavily_error
        raise SearchConfigurationError(
            "TAVILY_API_KEY or BRAVE_SEARCH_API_KEY is required for web search"
        )

    def _search_tavily(
        self,
        query: str,
        *,
        max_results: int,
        include_domains: tuple[str, ...] | None,
        official_only: bool,
    ) -> SearchResponse:
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if include_domains is not None:
            payload["include_domains"] = list(include_domains)
        try:
            response = self.http_client.post(
                TAVILY_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {self.tavily_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise SearchServiceError("Tavily request failed") from exc
        results = self._parse_tavily_results(
            data, max_results=max_results, official_only=official_only
        )
        return SearchResponse(query=query, provider="tavily", results=results)

    def _search_brave(self, query: str, *, max_results: int) -> SearchResponse:
        try:
            response = self.http_client.get(
                BRAVE_ENDPOINT,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": str(self.brave_api_key),
                },
                params={
                    "q": query,
                    "count": max_results,
                    "safesearch": "moderate",
                    "extra_snippets": "false",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise SearchServiceError("Brave request failed") from exc
        results = self._parse_brave_results(data, max_results=max_results)
        return SearchResponse(query=query, provider="brave", results=results)

    def _parse_tavily_results(
        self, data: object, *, max_results: int, official_only: bool
    ) -> tuple[SearchResult, ...]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchResponseError("Tavily response must contain a results array")
        return self._sanitize_results(
            data["results"],
            max_results=max_results,
            official_only=official_only,
            title_key="title",
            snippet_key="content",
            published_keys=("published_date", "published_at"),
        )

    def _parse_brave_results(
        self, data: object, *, max_results: int
    ) -> tuple[SearchResult, ...]:
        if not isinstance(data, dict) or not isinstance(data.get("web"), dict):
            raise SearchResponseError("Brave response must contain a web object")
        raw_results = data["web"].get("results")
        if not isinstance(raw_results, list):
            raise SearchResponseError("Brave response must contain web.results")
        return self._sanitize_results(
            raw_results,
            max_results=max_results,
            official_only=False,
            title_key="title",
            snippet_key="description",
            published_keys=("published_at", "age"),
        )

    def _sanitize_results(
        self,
        raw_results: list[object],
        *,
        max_results: int,
        official_only: bool,
        title_key: str,
        snippet_key: str,
        published_keys: tuple[str, ...],
    ) -> tuple[SearchResult, ...]:
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if len(results) >= max_results:
                break
            if not isinstance(item, dict):
                continue
            url = _safe_url(item.get("url"))
            if url is None or url in seen_urls:
                continue
            matched_source = self.registry.match_source(url)
            if official_only and matched_source is None:
                continue
            title = _clean_text(item.get(title_key), max_length=MAX_TITLE_CHARS)
            snippet = _clean_text(
                item.get(snippet_key), max_length=MAX_SNIPPET_CHARS
            )
            if not title or not snippet:
                continue
            published_at: str | None = None
            for key in published_keys:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    published_at = _clean_text(value, max_length=100)
                    break
            raw_score = item.get("score")
            score = (
                float(raw_score)
                if not isinstance(raw_score, bool)
                and isinstance(raw_score, (int, float))
                else None
            )
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_at=published_at,
                    score=score,
                    official_source=(
                        matched_source.name if matched_source is not None else None
                    ),
                    trust_level=(
                        matched_source.trust_level
                        if matched_source is not None
                        else None
                    ),
                )
            )
            seen_urls.add(url)
        return tuple(results)


__all__ = [
    "BRAVE_ENDPOINT",
    "DEFAULT_REGISTRY_PATH",
    "MAX_RESULTS",
    "OfficialSource",
    "OfficialSourceRegistry",
    "SearchConfigurationError",
    "SearchResponse",
    "SearchResponseError",
    "SearchResult",
    "SearchService",
    "SearchServiceError",
    "SearchValidationError",
    "TAVILY_ENDPOINT",
]
