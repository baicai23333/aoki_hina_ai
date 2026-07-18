import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from grounding import UIArtifact
from message_artifacts import (
    MAX_ARTIFACT_JSON_BYTES,
    MAX_BATCH_MESSAGE_IDS,
    MessageArtifactValidationError,
    init_message_artifacts_schema,
    list_artifacts_for_messages,
    list_message_artifacts,
    save_message_artifact,
    save_ui_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


class MessageArtifactTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / f".test_message_artifacts_{uuid4().hex}.db"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO chat_history (username, type, content, timestamp) "
                "VALUES ('alice', 'ai', 'reply', '2026-07-16T00:00:00Z')"
            )
            conn.execute(
                "INSERT INTO chat_history (username, type, content, timestamp) "
                "VALUES ('alice', 'ai', 'second', '2026-07-16T00:01:00Z')"
            )
            conn.commit()
            init_message_artifacts_schema(conn)
            conn.commit()

    def tearDown(self):
        for suffix in ("", "-journal", "-shm", "-wal"):
            Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)

    def test_save_list_and_parent_delete_cascades(self):
        artifact_id = save_message_artifact(
            self.db_path,
            1,
            "source_cards",
            {"items": [{"title": "Official", "url": "https://example.com"}]},
        )

        artifacts = list_message_artifacts(self.db_path, 1)
        self.assertEqual([item.id for item in artifacts], [artifact_id])
        self.assertEqual(artifacts[0].payload["items"][0]["title"], "Official")

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("DELETE FROM chat_history WHERE id = 1")
            conn.commit()
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM message_artifacts").fetchone()[0],
                0,
            )

    def test_batch_ui_artifacts_are_atomic_and_ordered(self):
        ids = save_ui_artifacts(
            self.db_path,
            1,
            (
                UIArtifact("weather_card", {"temperature": 30}),
                UIArtifact("search_status", {"count": 2}),
            ),
        )

        self.assertEqual(len(ids), 2)
        self.assertEqual(
            [item.artifact_type for item in list_message_artifacts(self.db_path, 1)],
            ["weather_card", "search_status"],
        )

    def test_batch_read_deduplicates_ids_and_keeps_empty_messages(self):
        first_id = save_message_artifact(
            self.db_path, 1, "search_status", {"ok": True}
        )

        grouped = list_artifacts_for_messages(self.db_path, [2, 1, 1])

        self.assertEqual(list(grouped), [2, 1])
        self.assertEqual(grouped[2], [])
        self.assertEqual([item.id for item in grouped[1]], [first_id])
        self.assertEqual(list_artifacts_for_messages(self.db_path, []), {})

    def test_batch_read_validates_ids_and_limit(self):
        with self.assertRaises(MessageArtifactValidationError):
            list_artifacts_for_messages(self.db_path, [1, 0])
        with self.assertRaises(MessageArtifactValidationError):
            list_artifacts_for_messages(self.db_path, "1")
        with self.assertRaises(MessageArtifactValidationError):
            list_artifacts_for_messages(
                self.db_path, range(1, MAX_BATCH_MESSAGE_IDS + 2)
            )

    def test_type_json_and_size_whitelists_are_enforced(self):
        with self.assertRaises(MessageArtifactValidationError):
            save_message_artifact(self.db_path, 1, "raw_html", {"html": "<b>x</b>"})
        with self.assertRaises(MessageArtifactValidationError):
            save_message_artifact(
                self.db_path, 1, "weather_card", {"unsupported": {1, 2, 3}}
            )
        with self.assertRaises(MessageArtifactValidationError):
            save_message_artifact(
                self.db_path,
                1,
                "weather_card",
                {"oversized": "x" * (MAX_ARTIFACT_JSON_BYTES + 1)},
            )
        with self.assertRaises(MessageArtifactValidationError):
            save_message_artifact(self.db_path, 999, "search_status", {"ok": True})

    def test_schema_contains_required_cascade_foreign_key(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("PRAGMA foreign_key_list(message_artifacts)").fetchall()

        self.assertTrue(
            any(
                row[2] == "chat_history"
                and row[3] == "message_id"
                and row[4] == "id"
                and row[6].upper() == "CASCADE"
                for row in rows
            )
        )


if __name__ == "__main__":
    unittest.main()
