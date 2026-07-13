from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from persona_pipeline import (
    EvidenceStore,
    FactStore,
    Intent,
    IntentClassifier,
    PersonaConfigurationError,
    SourceRegistry,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PERSONA_DIR = ROOT / "persona"
DEFAULT_CASES_PATH = DEFAULT_PERSONA_DIR / "evaluation_cases.jsonl"

DIMENSIONS = ("routing", "facts", "style", "sources", "boundary")
BOUNDARY_ACTIONS = {
    "none",
    "clarify_identity",
    "refuse_private",
    "insufficient_public_evidence",
}


class EvaluationCaseError(ValueError):
    """Raised when an offline evaluation case does not match the expected schema."""


def _required_text(data: dict[str, Any], field_name: str, context: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise EvaluationCaseError(f"{context}.{field_name} must be a non-empty string")
    return value.strip()


def _optional_string_set(
    data: dict[str, Any],
    field_name: str,
    context: str,
    *,
    allow_empty: bool = True,
) -> frozenset[str] | None:
    if field_name not in data:
        return None
    value = data[field_name]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise EvaluationCaseError(f"{context}.{field_name} must be an array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvaluationCaseError(
            f"{context}.{field_name} must contain only non-empty strings"
        )
    normalized = [item.strip() for item in value]
    if not allow_empty and not normalized:
        raise EvaluationCaseError(f"{context}.{field_name} cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise EvaluationCaseError(f"{context}.{field_name} cannot contain duplicates")
    return frozenset(normalized)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    input: str
    expected_intent: Intent
    tags: tuple[str, ...] = ()
    expected_fact_ids: frozenset[str] | None = None
    required_fact_ids: frozenset[str] | None = None
    forbidden_fact_ids: frozenset[str] | None = None
    required_evidence_ids: frozenset[str] | None = None
    forbidden_evidence_ids: frozenset[str] | None = None
    required_source_ids: frozenset[str] | None = None
    forbidden_source_ids: frozenset[str] | None = None
    expected_boundary_action: str | None = None

    ALLOWED_FIELDS = frozenset(
        {
            "id",
            "input",
            "expected_intent",
            "tags",
            "expected_fact_ids",
            "required_fact_ids",
            "forbidden_fact_ids",
            "required_evidence_ids",
            "forbidden_evidence_ids",
            "required_source_ids",
            "forbidden_source_ids",
            "expected_boundary_action",
        }
    )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        context: str = "evaluation case",
    ) -> "EvaluationCase":
        if not isinstance(data, dict):
            raise EvaluationCaseError(f"{context} must be a JSON object")
        unknown_fields = set(data) - cls.ALLOWED_FIELDS
        if unknown_fields:
            raise EvaluationCaseError(
                f"{context} contains unknown fields: {', '.join(sorted(unknown_fields))}"
            )

        expected_intent_text = _required_text(data, "expected_intent", context)
        try:
            expected_intent = Intent(expected_intent_text)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in Intent)
            raise EvaluationCaseError(
                f"{context}.expected_intent must be one of: {allowed}"
            ) from exc

        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or any(
            not isinstance(tag, str) or not tag.strip() for tag in raw_tags
        ):
            raise EvaluationCaseError(
                f"{context}.tags must be an array of non-empty strings"
            )

        expected_facts = _optional_string_set(data, "expected_fact_ids", context)
        required_facts = _optional_string_set(
            data, "required_fact_ids", context, allow_empty=False
        )
        forbidden_facts = _optional_string_set(
            data, "forbidden_fact_ids", context, allow_empty=False
        )
        required_evidence = _optional_string_set(
            data, "required_evidence_ids", context, allow_empty=False
        )
        forbidden_evidence = _optional_string_set(
            data, "forbidden_evidence_ids", context, allow_empty=False
        )
        required_sources = _optional_string_set(
            data, "required_source_ids", context, allow_empty=False
        )
        forbidden_sources = _optional_string_set(
            data, "forbidden_source_ids", context, allow_empty=False
        )
        for label, required, forbidden in (
            ("fact", required_facts, forbidden_facts),
            ("evidence", required_evidence, forbidden_evidence),
            ("source", required_sources, forbidden_sources),
        ):
            overlap = (required or frozenset()) & (forbidden or frozenset())
            if overlap:
                joined = ", ".join(sorted(overlap))
                raise EvaluationCaseError(
                    f"{context} requires and forbids the same {label} ids: {joined}"
                )
        if expected_facts is not None and required_facts is not None:
            raise EvaluationCaseError(
                f"{context} cannot declare both expected_fact_ids and required_fact_ids"
            )

        boundary = data.get("expected_boundary_action")
        if boundary is not None:
            if not isinstance(boundary, str) or boundary not in BOUNDARY_ACTIONS:
                raise EvaluationCaseError(
                    f"{context}.expected_boundary_action must be one of: "
                    f"{', '.join(sorted(BOUNDARY_ACTIONS))}"
                )

        return cls(
            case_id=_required_text(data, "id", context),
            input=_required_text(data, "input", context),
            expected_intent=expected_intent,
            tags=tuple(tag.strip() for tag in raw_tags),
            expected_fact_ids=expected_facts,
            required_fact_ids=required_facts,
            forbidden_fact_ids=forbidden_facts,
            required_evidence_ids=required_evidence,
            forbidden_evidence_ids=forbidden_evidence,
            required_source_ids=required_sources,
            forbidden_source_ids=forbidden_sources,
            expected_boundary_action=boundary,
        )


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load JSONL cases while reporting the exact line for malformed data."""

    if not path.exists():
        raise EvaluationCaseError(f"Evaluation case file is missing: {path}")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            context = f"{path}:{line_number}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationCaseError(
                    f"Invalid JSON at {context}: {exc.msg}"
                ) from exc
            case = EvaluationCase.from_dict(data, context)
            if case.case_id in seen_ids:
                raise EvaluationCaseError(
                    f"Duplicate evaluation case id at {context}: {case.case_id}"
                )
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise EvaluationCaseError(f"Evaluation case file contains no cases: {path}")
    return cases


@dataclass(frozen=True)
class CheckResult:
    dimension: str
    name: str
    passed: bool
    expected: Any
    actual: Any
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "name": self.name,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    input: str
    tags: tuple[str, ...]
    actual_intent: str
    fact_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    boundary_action: str
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "input": self.input,
            "tags": list(self.tags),
            "passed": self.passed,
            "actual_intent": self.actual_intent,
            "fact_ids": list(self.fact_ids),
            "evidence_ids": list(self.evidence_ids),
            "source_ids": list(self.source_ids),
            "boundary_action": self.boundary_action,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class DimensionScore:
    passed: int
    total: int

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.total else None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "total": self.total, "rate": self.rate}


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[CaseResult, ...]

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def case_pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases else 0.0

    @property
    def total_checks(self) -> int:
        return sum(len(result.checks) for result in self.results)

    @property
    def passed_checks(self) -> int:
        return sum(
            check.passed for result in self.results for check in result.checks
        )

    @property
    def overall_check_rate(self) -> float:
        return self.passed_checks / self.total_checks if self.total_checks else 0.0

    @property
    def dimension_scores(self) -> dict[str, DimensionScore]:
        scores: dict[str, DimensionScore] = {}
        for dimension in DIMENSIONS:
            checks = [
                check
                for result in self.results
                for check in result.checks
                if check.dimension == dimension
            ]
            scores[dimension] = DimensionScore(
                passed=sum(check.passed for check in checks),
                total=len(checks),
            )
        return scores

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "failed_cases": self.failed_cases,
                "case_pass_rate": self.case_pass_rate,
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_checks": self.total_checks - self.passed_checks,
                "overall_check_rate": self.overall_check_rate,
            },
            "dimensions": {
                name: score.to_dict()
                for name, score in self.dimension_scores.items()
            },
            "cases": [result.to_dict() for result in self.results],
        }


class PersonaEvaluator:
    """Deterministically evaluate routing and retrieval without invoking an LLM."""

    def __init__(
        self,
        source_registry: SourceRegistry,
        fact_store: FactStore,
        evidence_store: EvidenceStore,
        classifier: IntentClassifier | None = None,
    ):
        self.source_registry = source_registry
        self.fact_store = fact_store
        self.evidence_store = evidence_store
        self.classifier = classifier or IntentClassifier()

    @classmethod
    def from_persona_dir(cls, persona_dir: Path) -> "PersonaEvaluator":
        source_registry = SourceRegistry.from_jsonl(
            persona_dir / "source_registry.jsonl"
        )
        fact_store = FactStore.from_jsonl(
            persona_dir / "fact_claims.jsonl", source_registry
        )
        evidence_store = EvidenceStore.from_jsonl_paths(
            (
                persona_dir / "evidence_cards.jsonl",
                persona_dir / "style_evidence_cards.jsonl",
            ),
            source_registry,
        )
        return cls(source_registry, fact_store, evidence_store)

    @staticmethod
    def boundary_action(intent: Intent, has_facts: bool) -> str:
        if intent == Intent.IDENTITY_ATTACK:
            return "clarify_identity"
        if intent == Intent.PRIVATE_PROBE:
            return "refuse_private"
        if intent == Intent.PUBLIC_FACT and not has_facts:
            return "insufficient_public_evidence"
        return "none"

    def _validate_case_references(self, case: EvaluationCase) -> None:
        known_fact_ids = {
            claim.claim_id for claim in self.fact_store.claims
        } | set(self.fact_store.quarantined)
        known_evidence_ids = {
            card.card_id for card in self.evidence_store.cards
        } | set(self.evidence_store.quarantined)
        known_source_ids = set(self.source_registry.records)
        for field_name, declared, known_ids in (
            ("expected_fact_ids", case.expected_fact_ids, known_fact_ids),
            ("required_fact_ids", case.required_fact_ids, known_fact_ids),
            ("forbidden_fact_ids", case.forbidden_fact_ids, known_fact_ids),
            (
                "required_evidence_ids",
                case.required_evidence_ids,
                known_evidence_ids,
            ),
            (
                "forbidden_evidence_ids",
                case.forbidden_evidence_ids,
                known_evidence_ids,
            ),
            ("required_source_ids", case.required_source_ids, known_source_ids),
            ("forbidden_source_ids", case.forbidden_source_ids, known_source_ids),
        ):
            if declared is None:
                continue
            unknown = set(declared) - known_ids
            if unknown:
                raise EvaluationCaseError(
                    f"evaluation case {case.case_id}.{field_name} uses unknown ids: "
                    f"{', '.join(sorted(unknown))}"
                )

    def evaluate_case(self, case: EvaluationCase) -> CaseResult:
        self._validate_case_references(case)
        intent = self.classifier.classify(case.input)
        facts = (
            self.fact_store.retrieve(case.input)
            if intent == Intent.PUBLIC_FACT
            else []
        )
        evidence = self.evidence_store.retrieve(case.input, intent)
        fact_ids = tuple(claim.claim_id for claim in facts)
        evidence_ids = tuple(card.card_id for card in evidence)
        fact_id_set = set(fact_ids)
        evidence_id_set = set(evidence_ids)
        source_id_set = {
            citation.source_id for claim in facts for citation in claim.citations
        } | {source_id for card in evidence for source_id in card.source_ids}
        boundary_action = self.boundary_action(intent, bool(facts))

        checks: list[CheckResult] = []
        routing_passed = intent == case.expected_intent
        checks.append(
            CheckResult(
                dimension="routing",
                name="expected_intent",
                passed=routing_passed,
                expected=case.expected_intent.value,
                actual=intent.value,
                detail="" if routing_passed else "intent mismatch",
            )
        )

        if case.expected_fact_ids is not None:
            facts_passed = fact_id_set == set(case.expected_fact_ids)
            checks.append(
                CheckResult(
                    dimension="facts",
                    name="expected_fact_ids",
                    passed=facts_passed,
                    expected=sorted(case.expected_fact_ids),
                    actual=sorted(fact_id_set),
                    detail="" if facts_passed else "fact ids must match exactly",
                )
            )

        if case.required_fact_ids is not None:
            missing = set(case.required_fact_ids) - fact_id_set
            facts_passed = not missing
            checks.append(
                CheckResult(
                    dimension="facts",
                    name="required_fact_ids",
                    passed=facts_passed,
                    expected=sorted(case.required_fact_ids),
                    actual=sorted(fact_id_set),
                    detail=(
                        ""
                        if facts_passed
                        else f"missing required facts: {', '.join(sorted(missing))}"
                    ),
                )
            )

        if case.forbidden_fact_ids is not None:
            present = set(case.forbidden_fact_ids) & fact_id_set
            facts_passed = not present
            checks.append(
                CheckResult(
                    dimension="facts",
                    name="forbidden_fact_ids",
                    passed=facts_passed,
                    expected=sorted(case.forbidden_fact_ids),
                    actual=sorted(fact_id_set),
                    detail=(
                        ""
                        if facts_passed
                        else f"forbidden facts present: {', '.join(sorted(present))}"
                    ),
                )
            )

        if case.required_evidence_ids is not None:
            missing = set(case.required_evidence_ids) - evidence_id_set
            style_passed = not missing
            checks.append(
                CheckResult(
                    dimension="style",
                    name="required_evidence_ids",
                    passed=style_passed,
                    expected=sorted(case.required_evidence_ids),
                    actual=sorted(evidence_id_set),
                    detail=(
                        ""
                        if style_passed
                        else f"missing required evidence: {', '.join(sorted(missing))}"
                    ),
                )
            )

        if case.forbidden_evidence_ids is not None:
            present = set(case.forbidden_evidence_ids) & evidence_id_set
            style_passed = not present
            checks.append(
                CheckResult(
                    dimension="style",
                    name="forbidden_evidence_ids",
                    passed=style_passed,
                    expected=sorted(case.forbidden_evidence_ids),
                    actual=sorted(evidence_id_set),
                    detail=(
                        ""
                        if style_passed
                        else f"forbidden evidence present: {', '.join(sorted(present))}"
                    ),
                )
            )

        if case.required_source_ids is not None:
            missing = set(case.required_source_ids) - source_id_set
            sources_passed = not missing
            checks.append(
                CheckResult(
                    dimension="sources",
                    name="required_source_ids",
                    passed=sources_passed,
                    expected=sorted(case.required_source_ids),
                    actual=sorted(source_id_set),
                    detail=(
                        ""
                        if sources_passed
                        else f"missing required sources: {', '.join(sorted(missing))}"
                    ),
                )
            )

        if case.forbidden_source_ids is not None:
            present = set(case.forbidden_source_ids) & source_id_set
            sources_passed = not present
            checks.append(
                CheckResult(
                    dimension="sources",
                    name="forbidden_source_ids",
                    passed=sources_passed,
                    expected=sorted(case.forbidden_source_ids),
                    actual=sorted(source_id_set),
                    detail=(
                        ""
                        if sources_passed
                        else f"forbidden sources present: {', '.join(sorted(present))}"
                    ),
                )
            )

        if case.expected_boundary_action is not None:
            boundary_passed = boundary_action == case.expected_boundary_action
            checks.append(
                CheckResult(
                    dimension="boundary",
                    name="expected_boundary_action",
                    passed=boundary_passed,
                    expected=case.expected_boundary_action,
                    actual=boundary_action,
                    detail="" if boundary_passed else "boundary action mismatch",
                )
            )

        return CaseResult(
            case_id=case.case_id,
            input=case.input,
            tags=case.tags,
            actual_intent=intent.value,
            fact_ids=fact_ids,
            evidence_ids=evidence_ids,
            source_ids=tuple(sorted(source_id_set)),
            boundary_action=boundary_action,
            checks=tuple(checks),
        )

    def evaluate(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        return EvaluationReport(tuple(self.evaluate_case(case) for case in cases))


def _format_rate(rate: float | None) -> str:
    return "not scored" if rate is None else f"{rate:.1%}"


def render_text_report(report: EvaluationReport) -> str:
    lines = [
        "Offline persona evaluation",
        (
            f"Cases: {report.passed_cases}/{report.total_cases} passed "
            f"({_format_rate(report.case_pass_rate)})"
        ),
        (
            f"Checks: {report.passed_checks}/{report.total_checks} passed "
            f"({_format_rate(report.overall_check_rate)})"
        ),
        "Dimensions:",
    ]
    for name, score in report.dimension_scores.items():
        lines.append(
            f"  {name}: {score.passed}/{score.total} ({_format_rate(score.rate)})"
        )

    if report.failed_cases:
        lines.append("Failures:")
        for result in report.results:
            if result.passed:
                continue
            lines.append(f"  {result.case_id}: {result.input}")
            for check in result.failed_checks:
                lines.append(
                    f"    [{check.dimension}/{check.name}] {check.detail}; "
                    f"expected={check.expected!r}, actual={check.actual!r}"
                )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic, offline persona routing/retrieval evaluation."
    )
    parser.add_argument(
        "cases",
        nargs="?",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"JSONL evaluation cases (default: {DEFAULT_CASES_PATH})",
    )
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=DEFAULT_PERSONA_DIR,
        help=f"Persona data directory (default: {DEFAULT_PERSONA_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete machine-readable report as JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
        evaluator = PersonaEvaluator.from_persona_dir(args.persona_dir)
        report = evaluator.evaluate(cases)
    except (EvaluationCaseError, PersonaConfigurationError, OSError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"Evaluation setup error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_text_report(report))
    return 1 if report.failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
