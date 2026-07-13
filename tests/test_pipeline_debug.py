import json
import unittest
from dataclasses import dataclass

from pipeline_debug import build_debug_trace
from persona_pipeline import Intent


@dataclass
class FakeResult:
    intent: object
    evidence_ids: list[object]
    fact_ids: list[object]
    memory_ids: list[object]
    plan: dict
    validation_issues: list[object]


class PipelineDebugTests(unittest.TestCase):
    def test_trace_has_a_strict_non_sensitive_allowlist(self):
        secret = "SECRET user input / API key / D:\\private\\voice.wav"
        result = FakeResult(
            intent=Intent.MUSIC_ADVICE,
            evidence_ids=["PEC-012", secret],
            fact_ids=["FACT-AH-BIRTHDAY-001"],
            memory_ids=[3, -1, True, "4"],
            plan={
                "boundary_action": "none",
                "user_need": secret,
                "response_plan": [secret],
            },
            validation_issues=[secret, "review_rejected_draft"],
        )

        trace = build_debug_trace(
            result,
            translation_status="validated",
            tts_status="succeeded",
            stage_duration_ms={"pipeline": 12.6, "translation": -4, "other": 999},
        )
        serialized = json.dumps(trace, ensure_ascii=False)

        self.assertEqual(
            set(trace),
            {
                "intent",
                "route",
                "evidence_ids",
                "fact_ids",
                "memory_ids",
                "boundary_action",
                "validation_codes",
                "translation_status",
                "tts_status",
                "stage_duration_ms",
            },
        )
        self.assertEqual(trace["evidence_ids"], ["PEC-012"])
        self.assertEqual(trace["memory_ids"], [3])
        self.assertEqual(
            trace["validation_codes"],
            ["model_review_rejected", "review_rejected_draft"],
        )
        self.assertEqual(trace["stage_duration_ms"], {"pipeline": 13, "translation": 0})
        self.assertNotIn(secret, serialized)
        self.assertNotIn("user_need", serialized)
        self.assertNotIn("response_plan", serialized)

    def test_unknown_values_fail_to_safe_enumerations(self):
        result = FakeResult("../../secret", [], [], [], {}, [])

        trace = build_debug_trace(
            result,
            translation_status="raw_exception_text",
            tts_status="C:\\outside.wav",
        )

        self.assertEqual(trace["intent"], "unknown")
        self.assertEqual(trace["translation_status"], "failed")
        self.assertEqual(trace["tts_status"], "failed")


if __name__ == "__main__":
    unittest.main()
