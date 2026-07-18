"""Typed grounding shared by tools, storage, and presentation layers.

Web snippets remain explicitly untrusted data.  A ``GroundingBundle`` carries
facts, their source references, and optional UI artifacts without turning any
of them into prompt instructions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class GroundingValidationError(ValueError):
    """Raised when a grounding object violates its public contract."""


def _required_text(value: object, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise GroundingValidationError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise GroundingValidationError(f"{field_name} cannot be empty")
    if len(cleaned) > max_length:
        raise GroundingValidationError(
            f"{field_name} cannot exceed {max_length} characters"
        )
    return cleaned


@dataclass(frozen=True)
class GroundingSource:
    """One source that supports one or more facts."""

    id: str
    title: str
    url: str
    provider: str
    published_at: str | None = None
    trust_level: int | None = None
    untrusted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "id", max_length=80))
        object.__setattr__(
            self, "title", _required_text(self.title, "title", max_length=300)
        )
        object.__setattr__(self, "url", _required_text(self.url, "url", max_length=2048))
        object.__setattr__(
            self, "provider", _required_text(self.provider, "provider", max_length=80)
        )
        if self.published_at is not None and not isinstance(self.published_at, str):
            raise GroundingValidationError("published_at must be a string or None")
        if self.trust_level is not None:
            if (
                isinstance(self.trust_level, bool)
                or not isinstance(self.trust_level, int)
                or not 0 <= self.trust_level <= 100
            ):
                raise GroundingValidationError(
                    "trust_level must be an integer between 0 and 100 or None"
                )
        if not isinstance(self.untrusted, bool):
            raise GroundingValidationError("untrusted must be a boolean")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "provider": self.provider,
            "published_at": self.published_at,
            "trust_level": self.trust_level,
            "untrusted": self.untrusted,
        }


@dataclass(frozen=True)
class GroundedFact:
    """A bounded statement with optional source references."""

    text: str
    source_ids: tuple[str, ...] = ()
    confidence: float | None = None
    untrusted: bool = False
    fact_eligible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "text", _required_text(self.text, "text", max_length=4_000)
        )
        if not isinstance(self.source_ids, tuple):
            object.__setattr__(self, "source_ids", tuple(self.source_ids))
        normalized_ids: list[str] = []
        for source_id in self.source_ids:
            normalized_ids.append(
                _required_text(source_id, "source_id", max_length=80)
            )
        object.__setattr__(self, "source_ids", tuple(normalized_ids))
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise GroundingValidationError("confidence must be numeric or None")
            confidence = float(self.confidence)
            if not 0.0 <= confidence <= 1.0:
                raise GroundingValidationError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.untrusted, bool):
            raise GroundingValidationError("untrusted must be a boolean")
        if not isinstance(self.fact_eligible, bool):
            raise GroundingValidationError("fact_eligible must be a boolean")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "text": self.text,
            "source_ids": list(self.source_ids),
            "confidence": self.confidence,
            "untrusted": self.untrusted,
            "fact_eligible": self.fact_eligible,
        }


@dataclass(frozen=True)
class UIArtifact:
    """Presentation metadata; never interpreted as model instructions."""

    artifact_type: str
    payload: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_type",
            _required_text(self.artifact_type, "artifact_type", max_length=80),
        )
        if not isinstance(self.payload, Mapping):
            raise GroundingValidationError("payload must be a mapping")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_dict(self) -> dict[str, JSONValue]:
        return {"artifact_type": self.artifact_type, "payload": dict(self.payload)}


@dataclass(frozen=True)
class GroundingBundle:
    """All non-conversational evidence produced during one tool run."""

    facts: tuple[GroundedFact, ...] = field(default_factory=tuple)
    sources: tuple[GroundingSource, ...] = field(default_factory=tuple)
    ui_artifacts: tuple[UIArtifact, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.facts, tuple):
            object.__setattr__(self, "facts", tuple(self.facts))
        if not isinstance(self.sources, tuple):
            object.__setattr__(self, "sources", tuple(self.sources))
        if not isinstance(self.ui_artifacts, tuple):
            object.__setattr__(self, "ui_artifacts", tuple(self.ui_artifacts))
        if not all(isinstance(item, GroundedFact) for item in self.facts):
            raise GroundingValidationError("facts must contain GroundedFact values")
        if not all(isinstance(item, GroundingSource) for item in self.sources):
            raise GroundingValidationError(
                "sources must contain GroundingSource values"
            )
        if not all(isinstance(item, UIArtifact) for item in self.ui_artifacts):
            raise GroundingValidationError(
                "ui_artifacts must contain UIArtifact values"
            )
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise GroundingValidationError("source ids must be unique within a bundle")
        known_ids = set(source_ids)
        for fact in self.facts:
            unknown = set(fact.source_ids) - known_ids
            if unknown:
                raise GroundingValidationError(
                    f"fact references unknown source ids: {sorted(unknown)!r}"
                )

    @classmethod
    def empty(cls) -> "GroundingBundle":
        return cls()

    @classmethod
    def from_search_response(cls, response: Any) -> "GroundingBundle":
        """Convert a typed search response without trusting its snippets."""

        provider = _required_text(
            getattr(response, "provider", "search"), "provider", max_length=80
        )
        results = getattr(response, "results", ())
        if not isinstance(results, (tuple, list)):
            raise GroundingValidationError("search response results must be a sequence")

        facts: list[GroundedFact] = []
        sources: list[GroundingSource] = []
        cards: list[JSONValue] = []
        fact_eligible = getattr(response, "fact_eligible", False) is True
        for index, result in enumerate(results, start=1):
            source_id = f"search-{index}"
            title = _required_text(
                getattr(result, "title", "Search result"),
                "search title",
                max_length=300,
            )
            url = _required_text(
                getattr(result, "url", ""), "search url", max_length=2048
            )
            snippet_value = getattr(result, "snippet", "")
            snippet = snippet_value.strip() if isinstance(snippet_value, str) else ""
            published_at = getattr(result, "published_at", None)
            trust_value = getattr(result, "trust_level", None)
            trust_level = (
                trust_value
                if not isinstance(trust_value, bool)
                and isinstance(trust_value, int)
                and 0 <= trust_value <= 100
                else None
            )
            official_source_value = getattr(result, "official_source", None)
            official_source = (
                official_source_value.strip()[:80]
                if isinstance(official_source_value, str)
                and official_source_value.strip()
                else None
            )
            sources.append(
                GroundingSource(
                    id=source_id,
                    title=title,
                    url=url,
                    provider=provider,
                    published_at=published_at if isinstance(published_at, str) else None,
                    trust_level=trust_level,
                    untrusted=True,
                )
            )
            facts.append(
                GroundedFact(
                    text=snippet or title,
                    source_ids=(source_id,),
                    untrusted=True,
                    fact_eligible=fact_eligible,
                )
            )
            cards.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "published_at": (
                        published_at if isinstance(published_at, str) else None
                    ),
                    "provider": provider,
                    "official_source": official_source,
                    "trust_level": trust_level,
                    "untrusted": True,
                }
            )
        artifacts = (
            UIArtifact(
                artifact_type="source_cards",
                payload={"items": cards, "untrusted": True},
            ),
        ) if cards else ()
        return cls(tuple(facts), tuple(sources), artifacts)

    def merge(self, *others: "GroundingBundle") -> "GroundingBundle":
        """Merge bundles while assigning collision-free source identifiers."""

        facts = list(self.facts)
        sources = list(self.sources)
        artifacts = list(self.ui_artifacts)
        known_ids = {source.id for source in sources}
        for bundle_index, other in enumerate(others, start=1):
            if not isinstance(other, GroundingBundle):
                raise TypeError("merge expects GroundingBundle values")
            id_map: dict[str, str] = {}
            for source in other.sources:
                candidate = source.id
                suffix = 1
                while candidate in known_ids:
                    candidate = f"{source.id}-{bundle_index}-{suffix}"
                    suffix += 1
                known_ids.add(candidate)
                id_map[source.id] = candidate
                sources.append(
                    GroundingSource(
                        id=candidate,
                        title=source.title,
                        url=source.url,
                        provider=source.provider,
                        published_at=source.published_at,
                        trust_level=source.trust_level,
                        untrusted=source.untrusted,
                    )
                )
            for fact in other.facts:
                facts.append(
                    GroundedFact(
                        text=fact.text,
                        source_ids=tuple(id_map[item] for item in fact.source_ids),
                        confidence=fact.confidence,
                        untrusted=fact.untrusted,
                        fact_eligible=fact.fact_eligible,
                    )
                )
            artifacts.extend(other.ui_artifacts)
        return GroundingBundle(tuple(facts), tuple(sources), tuple(artifacts))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "facts": [fact.to_dict() for fact in self.facts],
            "sources": [source.to_dict() for source in self.sources],
            "ui_artifacts": [artifact.to_dict() for artifact in self.ui_artifacts],
        }

    def to_tool_json(self, *, max_bytes: int = 16_384) -> str:
        """Serialize a bounded tool payload without cutting JSON mid-token."""

        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 256:
            raise GroundingValidationError("max_bytes must be an integer of at least 256")
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if len(encoded.encode("utf-8")) <= max_bytes:
            return encoded
        compact = {
            "facts": [],
            "sources": [
                {
                    "id": source.id,
                    "title": source.title,
                    "url": source.url,
                    "untrusted": source.untrusted,
                }
                for source in self.sources[:5]
            ],
            "ui_artifacts": [],
            "truncated": True,
        }
        encoded = json.dumps(
            compact, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if len(encoded.encode("utf-8")) > max_bytes:
            return '{"facts":[],"sources":[],"ui_artifacts":[],"truncated":true}'
        return encoded


__all__ = [
    "GroundedFact",
    "GroundingBundle",
    "GroundingSource",
    "GroundingValidationError",
    "JSONValue",
    "UIArtifact",
]
