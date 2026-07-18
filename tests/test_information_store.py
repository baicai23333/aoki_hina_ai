from __future__ import annotations

import sqlite3
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from information_store import (
    InformationSchemaError,
    InformationValidationError,
    add_information_source,
    content_sha256,
    create_pending_update,
    finish_collector_run,
    get_information_source,
    init_information_schema,
    list_collector_runs,
    list_official_updates,
    query_recent_approved_updates,
    revoke_official_update,
    review_official_update,
    set_source_enabled,
    start_collector_run,
    store_source_document,
)


ROOT = Path(__file__).resolve().parents[1]


class InformationStoreTests(unittest.TestCase):
    def setUp(self):
        self.db_path = ROOT / f".test_information_store_{uuid4().hex}.db"

    def tearDown(self):
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)

    def add_source(self):
        return add_information_source(
            self.db_path,
            name="Official News",
            source_type="rss",
            base_url="https://official.example/news/?utm_source=test",
            allowed_domains=["media.official.example"],
            trust_level=100,
            fetch_interval_minutes=30,
        )

    def create_admin_audit_table(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE admin_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    actor TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_username TEXT,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def test_schema_is_additive_idempotent_and_preserves_unrelated_tables(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE legacy_data (value TEXT)")
            conn.execute("INSERT INTO legacy_data VALUES ('keep')")
            conn.commit()
            init_information_schema(conn)
            init_information_schema(conn)
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5_000)
            conn.commit()
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertTrue(
                {
                    "legacy_data",
                    "information_sources",
                    "source_documents",
                    "official_updates",
                    "collector_runs",
                }.issubset(tables)
            )
            self.assertEqual(conn.execute("SELECT value FROM legacy_data").fetchone()[0], "keep")
            conn.commit()
            init_information_schema(conn)
            self.assertFalse(conn.in_transaction)

    def test_incompatible_existing_table_fails_before_partial_creation(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE information_sources (id INTEGER PRIMARY KEY)")
            conn.commit()
            with self.assertRaises(InformationSchemaError):
                init_information_schema(conn)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            self.assertNotIn("source_documents", tables)
            self.assertNotIn("official_updates", tables)
            self.assertNotIn("collector_runs", tables)

    def test_sources_are_canonicalized_validated_and_uniquely_scoped(self):
        source = self.add_source()
        self.assertEqual(source.base_url, "https://official.example/news/")
        self.assertEqual(
            source.allowed_domains,
            ("media.official.example", "official.example"),
        )
        self.assertTrue(source.enabled)
        with self.assertRaises(InformationValidationError):
            self.add_source()
        with self.assertRaises(InformationValidationError):
            add_information_source(
                self.db_path,
                name="Bad",
                source_type="social",
                base_url="https://official.example/other",
            )

    def test_document_hash_deduplicates_before_new_rows(self):
        source = self.add_source()
        content = "Official announcement body"
        first, created = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/1?utm_campaign=x",
            title="Announcement",
            raw_content=content,
            content_hash=content_sha256(content),
            fetched_at="2026-07-16T00:00:00Z",
        )
        second, created_again = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/duplicate",
            raw_content=content,
            fetched_at="2026-07-16T01:00:00Z",
        )
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.canonical_url, "https://official.example/news/1")
        self.assertEqual(second.fetched_at, "2026-07-16T01:00:00.000000Z")

    def test_pending_is_excluded_until_atomic_admin_approval(self):
        source = self.add_source()
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        document, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/live-1",
            raw_content="A new official live event",
            published_at=now,
        )
        update, created = create_pending_update(
            self.db_path,
            document_id=document.id,
            category="live",
            title="Official Live",
            summary="Confirmed announcement",
            event_start_at=now + timedelta(days=10),
            venue="Tokyo",
            confidence=0.98,
        )
        self.assertTrue(created)
        self.assertEqual(update.verification_status, "pending")
        self.assertEqual(query_recent_approved_updates(self.db_path, now=now), [])

        self.create_admin_audit_table()
        approved = review_official_update(
            self.db_path, update.id, "approved", actor="admin", reason="official source"
        )
        self.assertEqual(approved.verification_status, "approved")
        recent = query_recent_approved_updates(self.db_path, now=now)
        self.assertEqual([item.id for item in recent], [update.id])
        with closing(sqlite3.connect(self.db_path)) as conn:
            audit = conn.execute(
                "SELECT actor, action, detail FROM admin_audit_log"
            ).fetchone()
        self.assertEqual(audit[0], "admin")
        self.assertEqual(audit[1], "information.update_approved")
        self.assertIn(f'"update_id":{update.id}', audit[2])
        with self.assertRaises(InformationValidationError):
            review_official_update(self.db_path, update.id, "rejected", actor="admin")

    def test_source_toggle_and_audit_insert_are_one_transaction(self):
        source = self.add_source()
        self.create_admin_audit_table()
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TRIGGER reject_information_audit
                BEFORE INSERT ON admin_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'simulated audit failure');
                END
                """
            )
        with self.assertRaises(sqlite3.IntegrityError):
            set_source_enabled(self.db_path, source.id, False, actor="admin")
        self.assertTrue(get_information_source(self.db_path, source.id).enabled)

    def test_approved_update_can_be_revoked_with_audit(self):
        source = self.add_source()
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        document, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/revoke",
            raw_content="Official notice later found invalid",
            published_at=now,
        )
        update, _ = create_pending_update(
            self.db_path,
            document_id=document.id,
            category="announcement",
            title="Withdrawn notice",
            confidence=0.99,
        )
        self.create_admin_audit_table()
        review_official_update(self.db_path, update.id, "approved", actor="reviewer")

        revoked = revoke_official_update(
            self.db_path,
            update.id,
            actor="admin",
            reason="official page was withdrawn",
        )

        self.assertEqual(revoked.verification_status, "rejected")
        self.assertEqual(query_recent_approved_updates(self.db_path, now=now), [])
        with closing(sqlite3.connect(self.db_path)) as conn:
            audit = conn.execute(
                "SELECT actor, action, detail FROM admin_audit_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        self.assertEqual(audit[0], "admin")
        self.assertEqual(audit[1], "information.update_revoked")
        self.assertIn("official page was withdrawn", audit[2])
        with self.assertRaises(InformationValidationError):
            revoke_official_update(
                self.db_path,
                update.id,
                actor="admin",
                reason="second attempt",
            )

    def test_replacement_relation_and_collector_run_lifecycle(self):
        source = self.add_source()
        first_doc, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/old",
            raw_content="Old event details",
        )
        first, _ = create_pending_update(
            self.db_path,
            document_id=first_doc.id,
            category="event",
            title="Old Event",
            confidence=0.9,
        )
        second_doc, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/correction",
            raw_content="Corrected event details",
        )
        second, _ = create_pending_update(
            self.db_path,
            document_id=second_doc.id,
            category="correction",
            title="Correction",
            status="updated",
            confidence=0.95,
            replaces_update_id=first.id,
        )
        self.assertEqual(second.replaces_update_id, first.id)

        run = start_collector_run(self.db_path, source.id)
        self.assertEqual(run.status, "running")
        finished = finish_collector_run(
            self.db_path,
            run.id,
            status="partial",
            discovered_count=3,
            fetched_count=2,
            new_document_count=1,
            pending_update_count=1,
            error_code="partial_document_failures",
            detail='{"error_codes":["fetch_failed"]}',
        )
        self.assertEqual(finished.status, "partial")
        self.assertEqual(finished.discovered_count, 3)
        with self.assertRaises(InformationValidationError):
            finish_collector_run(
                self.db_path,
                run.id,
                status="succeeded",
                discovered_count=0,
                fetched_count=0,
                new_document_count=0,
                pending_update_count=0,
            )

    def test_approved_replacement_excludes_the_old_update(self):
        source = self.add_source()
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        old_doc, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/event-old",
            raw_content="Original event date",
            published_at=now,
        )
        old_update, _ = create_pending_update(
            self.db_path,
            document_id=old_doc.id,
            category="event",
            title="Original event",
            event_start_at=now + timedelta(days=10),
            confidence=0.98,
        )
        replacement_doc, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/event-correction",
            raw_content="Corrected event date",
            published_at=now + timedelta(hours=1),
        )
        replacement, _ = create_pending_update(
            self.db_path,
            document_id=replacement_doc.id,
            category="correction",
            title="Corrected event",
            status="updated",
            event_start_at=now + timedelta(days=11),
            confidence=0.99,
            replaces_update_id=old_update.id,
        )
        latest_doc, _ = store_source_document(
            self.db_path,
            source_id=source.id,
            canonical_url="https://official.example/news/event-final",
            raw_content="Final corrected event date",
            published_at=now + timedelta(hours=2),
        )
        latest, _ = create_pending_update(
            self.db_path,
            document_id=latest_doc.id,
            category="correction",
            title="Final corrected event",
            status="updated",
            event_start_at=now + timedelta(days=12),
            confidence=0.99,
            replaces_update_id=replacement.id,
        )
        self.create_admin_audit_table()
        review_official_update(self.db_path, old_update.id, "approved", actor="admin")
        review_official_update(self.db_path, replacement.id, "approved", actor="admin")
        review_official_update(self.db_path, latest.id, "approved", actor="admin")
        revoke_official_update(
            self.db_path,
            replacement.id,
            actor="admin",
            reason="superseded by the final correction",
        )

        recent = query_recent_approved_updates(self.db_path, now=now)

        self.assertEqual([item.id for item in recent], [latest.id])

    def test_admin_read_lists_filter_and_return_newest_first_with_source_name(self):
        first_source = self.add_source()
        second_source = add_information_source(
            self.db_path,
            name="Second Official",
            source_type="html",
            base_url="https://second.example/news/",
        )
        first_doc, _ = store_source_document(
            self.db_path,
            source_id=first_source.id,
            canonical_url="https://official.example/news/first",
            raw_content="First pending update",
        )
        first_update, _ = create_pending_update(
            self.db_path,
            document_id=first_doc.id,
            category="announcement",
            title="First",
            confidence=0.9,
        )
        second_doc, _ = store_source_document(
            self.db_path,
            source_id=second_source.id,
            canonical_url="https://second.example/news/second",
            raw_content="Second approved update",
        )
        second_update, _ = create_pending_update(
            self.db_path,
            document_id=second_doc.id,
            category="event",
            title="Second",
            confidence=0.95,
        )
        review_official_update(
            self.db_path, second_update.id, "approved", actor="admin"
        )
        self.assertEqual(
            [item.id for item in list_official_updates(self.db_path)],
            [second_update.id, first_update.id],
        )
        self.assertEqual(
            [
                item.id
                for item in list_official_updates(
                    self.db_path, verification_status="pending"
                )
            ],
            [first_update.id],
        )

        first_run = start_collector_run(self.db_path, first_source.id)
        finish_collector_run(
            self.db_path,
            first_run.id,
            status="succeeded",
            discovered_count=0,
            fetched_count=0,
            new_document_count=0,
            pending_update_count=0,
        )
        second_run = start_collector_run(self.db_path, second_source.id)
        finish_collector_run(
            self.db_path,
            second_run.id,
            status="failed",
            discovered_count=0,
            fetched_count=0,
            new_document_count=0,
            pending_update_count=0,
            error_code="source_fetch_failed",
        )
        runs = list_collector_runs(self.db_path)
        self.assertEqual([item.id for item in runs], [second_run.id, first_run.id])
        self.assertEqual(
            [item.source_name for item in runs], ["Second Official", "Official News"]
        )
        filtered = list_collector_runs(
            self.db_path, source_id=first_source.id, limit=1
        )
        self.assertEqual([item.id for item in filtered], [first_run.id])


if __name__ == "__main__":
    unittest.main()
