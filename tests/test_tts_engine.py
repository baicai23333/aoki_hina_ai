import os
import shutil
import socket
import struct
import subprocess
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import requests

import tts_engine
from tts_engine import TTSConfig, TTSEngine, safe_cached_wav_path


ROOT = Path(__file__).resolve().parents[1]


def valid_wav_bytes(pcm=b"\x00\x00\x00\x00") -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8_000, 16_000, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


class TTSEngineHardeningTests(unittest.TestCase):
    def setUp(self):
        self.test_root = ROOT / f".test_tts_engine_{uuid4().hex}"
        self.cache_dir = self.test_root / "cache"
        self.cache_dir.mkdir(parents=True)
        self.reference_audio = self.test_root / "reference.wav"
        self.reference_audio.write_bytes(valid_wav_bytes())
        self.outside_files = []
        TTSEngine._instance = None

    def tearDown(self):
        TTSEngine._instance = None
        shutil.rmtree(self.test_root, ignore_errors=True)
        for path in self.outside_files:
            path.unlink(missing_ok=True)

    def make_config(
        self,
        *,
        normalize=False,
        output_language="ja",
        cache_max_files=tts_engine.DEFAULT_CACHE_MAX_FILES,
        cache_max_bytes=tts_engine.DEFAULT_CACHE_MAX_BYTES,
        api_url="http://127.0.0.1:9880/tts",
        auto_start=False,
    ) -> TTSConfig:
        return TTSConfig(
            backend="gpt_sovits",
            model_name="gpt-sovits-v2proplus",
            tts_home=self.test_root / "tts-home",
            cache_dir=self.cache_dir,
            max_chars=400,
            speaker_name="MyVoice",
            language="ja",
            xtts_use_cached_voice=False,
            xtts_refs_manifest=self.test_root / "refs.json",
            gpt_sovits_api_url=api_url,
            gpt_sovits_reference_audio=self.reference_audio,
            gpt_sovits_reference_text="reference text",
            gpt_sovits_prompt_language="ja",
            gpt_sovits_output_language=output_language,
            gpt_sovits_timeout=10,
            gpt_sovits_auto_start=auto_start,
            gpt_sovits_root=self.test_root / "gpt-sovits",
            gpt_sovits_python=self.test_root / "python.exe",
            gpt_sovits_config=self.test_root / "config.yaml",
            gpt_sovits_normalize_loudness=normalize,
            gpt_sovits_target_lufs=-16.0,
            cache_max_files=cache_max_files,
            cache_max_bytes=cache_max_bytes,
        )

    def test_get_defaults_gpt_sovits_output_language_to_ja(self):
        with patch.object(tts_engine, "load_env_var", side_effect=lambda key, default=None: default), patch.object(
            tts_engine, "DEFAULT_CACHE_DIR", self.cache_dir
        ):
            engine = TTSEngine.get()

        self.assertEqual(engine.config.gpt_sovits_output_language, "ja")
        self.assertEqual(engine.config.cache_max_files, tts_engine.DEFAULT_CACHE_MAX_FILES)
        self.assertEqual(engine.config.cache_max_bytes, tts_engine.DEFAULT_CACHE_MAX_BYTES)

    def test_gpt_sovits_rejects_non_japanese_output_language(self):
        with self.assertRaisesRegex(RuntimeError, "Japanese-compatible"):
            TTSEngine(self.make_config(output_language="zh"))

    def test_safe_cached_wav_path_accepts_only_valid_files_within_cache(self):
        valid_cached = self.cache_dir / "valid.wav"
        valid_cached.write_bytes(valid_wav_bytes())
        outside = ROOT / f".test_tts_outside_{uuid4().hex}.wav"
        outside.write_bytes(valid_wav_bytes())
        self.outside_files.append(outside)
        invalid_cached = self.cache_dir / "invalid.wav"
        invalid_cached.write_bytes(b"not a wav")

        self.assertEqual(safe_cached_wav_path(valid_cached, self.cache_dir), valid_cached.resolve())
        self.assertIsNone(safe_cached_wav_path(outside, self.cache_dir))
        self.assertIsNone(
            safe_cached_wav_path(self.cache_dir / ".." / self.reference_audio.name, self.cache_dir)
        )
        self.assertIsNone(safe_cached_wav_path(invalid_cached, self.cache_dir))

    def test_safe_cached_wav_path_uses_the_configured_cache_by_default(self):
        valid_cached = self.cache_dir / "configured.wav"
        valid_cached.write_bytes(valid_wav_bytes())

        with patch.object(
            tts_engine,
            "load_env_var",
            side_effect=lambda key, default=None: (
                str(self.cache_dir) if key == "AOKI_TTS_CACHE_DIR" else default
            ),
        ):
            resolved = safe_cached_wav_path(valid_cached)

        self.assertEqual(resolved, valid_cached.resolve())

    def test_invalid_http_200_responses_leave_no_audio_files(self):
        cases = (
            ("application/json", b'{"error": "bad request"}'),
            ("audio/wav", b"not a wav"),
        )
        for content_type, body in cases:
            with self.subTest(content_type=content_type, body=body):
                engine = TTSEngine(self.make_config())
                response = Mock()
                response.headers = {"Content-Type": content_type}
                response.content = body
                response.raise_for_status.return_value = None
                with patch.object(engine, "_ensure_gpt_sovits_api"), patch.object(
                    requests, "post", return_value=response
                ):
                    with self.assertRaisesRegex(RuntimeError, "invalid"):
                        engine.synthesize_to_file("こんにちは")

                self.assertEqual(list(self.cache_dir.glob("*.wav")), [])

    def test_normalization_failure_cleans_all_intermediate_audio(self):
        engine = TTSEngine(self.make_config(normalize=True))
        ffmpeg_path = engine.config.gpt_sovits_root / "runtime" / "ffmpeg.exe"
        ffmpeg_path.parent.mkdir(parents=True)
        ffmpeg_path.write_bytes(b"test stub")
        response = Mock()
        response.headers = {"Content-Type": "audio/wav; charset=binary"}
        response.content = valid_wav_bytes()
        response.raise_for_status.return_value = None

        def fail_normalization(command, **kwargs):
            Path(command[-1]).write_bytes(b"partial normalized output")
            raise subprocess.CalledProcessError(1, command, stderr=b"normalization failed")

        with patch.object(engine, "_ensure_gpt_sovits_api"), patch.object(
            requests, "post", return_value=response
        ), patch.object(subprocess, "run", side_effect=fail_normalization):
            with self.assertRaisesRegex(RuntimeError, "normalization failed"):
                engine.synthesize_to_file("こんにちは")

        self.assertEqual(list(self.cache_dir.glob("*.wav")), [])

    def test_wave_validation_rejects_magic_only_and_truncated_files(self):
        magic_only = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + (b"\x00" * 32)
        truncated = valid_wav_bytes()[:-2]
        invalid_path = self.cache_dir / "magic-only.wav"
        invalid_path.write_bytes(magic_only)

        self.assertFalse(tts_engine._is_valid_wav_bytes(magic_only))
        self.assertFalse(tts_engine._is_valid_wav_bytes(truncated))
        self.assertIsNone(safe_cached_wav_path(invalid_path, self.cache_dir))

    def test_startup_removes_stale_temp_wavs_but_keeps_final_cache(self):
        stale_raw = self.cache_dir / "stale.raw.wav"
        stale_normalized = self.cache_dir / "stale.normalized.wav"
        final_wav = self.cache_dir / "final.wav"
        for path in (stale_raw, stale_normalized, final_wav):
            path.write_bytes(valid_wav_bytes())

        TTSEngine(self.make_config())

        self.assertFalse(stale_raw.exists())
        self.assertFalse(stale_normalized.exists())
        self.assertTrue(final_wav.exists())

    def test_cache_key_changes_with_reference_audio_and_config_contents(self):
        engine = TTSEngine(self.make_config())
        original_key = engine._cache_key("same text")

        self.reference_audio.write_bytes(valid_wav_bytes(b"\x01\x00\x01\x00"))
        changed_reference_key = engine._cache_key("same text")
        engine.config.gpt_sovits_config.write_text("weights: first", encoding="utf-8")
        first_config_key = engine._cache_key("same text")
        engine.config.gpt_sovits_config.write_text("weights: second", encoding="utf-8")
        second_config_key = engine._cache_key("same text")

        self.assertNotEqual(original_key, changed_reference_key)
        self.assertNotEqual(changed_reference_key, first_config_key)
        self.assertNotEqual(first_config_key, second_config_key)

    def test_cache_prunes_oldest_files_before_and_after_synthesis(self):
        wav_size = len(valid_wav_bytes())
        older = self.cache_dir / "older.wav"
        newer = self.cache_dir / "newer.wav"
        older.write_bytes(valid_wav_bytes())
        newer.write_bytes(valid_wav_bytes())
        os.utime(older, ns=(1_000_000_000, 1_000_000_000))
        os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
        engine = TTSEngine(
            self.make_config(cache_max_files=2, cache_max_bytes=wav_size * 2)
        )
        response = Mock()
        response.headers = {"Content-Type": "audio/wav"}
        response.content = valid_wav_bytes()
        response.raise_for_status.return_value = None

        with patch.object(engine, "_ensure_gpt_sovits_api"), patch.object(
            requests, "post", return_value=response
        ):
            generated = engine.synthesize_to_file("cache pruning")

        final_files = [
            path
            for path in self.cache_dir.glob("*.wav")
            if not engine._is_temporary_wav(path)
        ]
        self.assertTrue(generated.exists())
        self.assertFalse(older.exists())
        self.assertLessEqual(len(final_files), 2)
        self.assertLessEqual(sum(path.stat().st_size for path in final_files), wav_size * 2)

    def test_oversized_audio_is_deleted_instead_of_cached(self):
        body = valid_wav_bytes()
        engine = TTSEngine(
            self.make_config(cache_max_files=5, cache_max_bytes=len(body) - 1)
        )
        response = Mock()
        response.headers = {"Content-Type": "audio/wav"}
        response.content = body
        response.raise_for_status.return_value = None

        with patch.object(engine, "_ensure_gpt_sovits_api"), patch.object(
            requests, "post", return_value=response
        ):
            with self.assertRaisesRegex(RuntimeError, "CACHE_MAX_BYTES"):
                engine.synthesize_to_file("too large")

        self.assertEqual(list(self.cache_dir.glob("*.wav")), [])

    def test_same_cache_key_concurrency_synthesizes_only_once(self):
        engine = TTSEngine(self.make_config())
        second_engine = TTSEngine(self.make_config())
        response = Mock()
        response.headers = {"Content-Type": "audio/wav"}
        response.content = valid_wav_bytes()
        response.raise_for_status.return_value = None
        first_request_started = threading.Event()
        release_first_request = threading.Event()
        second_call_started = threading.Event()

        def delayed_post(*args, **kwargs):
            first_request_started.set()
            self.assertTrue(release_first_request.wait(2))
            return response

        def synthesize_after_signal():
            second_call_started.set()
            return second_engine.synthesize_to_file("same concurrent text")

        with patch.object(engine, "_ensure_gpt_sovits_api"), patch.object(
            requests, "post", side_effect=delayed_post
        ) as post_mock, ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(engine.synthesize_to_file, "same concurrent text")
            self.assertTrue(first_request_started.wait(2))
            second = executor.submit(synthesize_after_signal)
            self.assertTrue(second_call_started.wait(2))
            release_first_request.set()
            first_path = first.result(timeout=2)
            second_path = second.result(timeout=2)

        self.assertEqual(first_path, second_path)
        self.assertEqual(post_mock.call_count, 1)
        self.assertEqual(list(self.cache_dir.glob("*.raw.wav")), [])
        self.assertEqual(list(self.cache_dir.glob("*.normalized.wav")), [])

    def test_temporary_wav_names_are_unique(self):
        engine = TTSEngine(self.make_config())
        final_path = self.cache_dir / "final.wav"

        first = engine._temporary_wav_path(final_path, "raw")
        second = engine._temporary_wav_path(final_path, "raw")

        self.assertNotEqual(first, second)
        self.assertTrue(first.name.endswith(".raw.wav"))

    def test_health_check_uses_host_and_port_from_api_url(self):
        engine = TTSEngine(
            self.make_config(api_url="http://localhost:9999/custom/tts")
        )
        connection = MagicMock()

        with patch.object(socket, "create_connection", return_value=connection) as connect_mock:
            self.assertTrue(engine._gpt_sovits_api_reachable())

        connect_mock.assert_called_once_with(("localhost", 9999), timeout=0.5)

    def test_remote_api_is_never_auto_started(self):
        engine = TTSEngine(
            self.make_config(api_url="https://example.com:9443/tts", auto_start=True)
        )

        with patch.object(engine, "_gpt_sovits_api_reachable", return_value=False), patch.object(
            subprocess, "Popen"
        ) as popen_mock:
            with self.assertRaisesRegex(RuntimeError, "only for a local HTTP"):
                engine._ensure_gpt_sovits_api()

        popen_mock.assert_not_called()

    def test_local_auto_start_uses_configured_port(self):
        config = self.make_config(
            api_url="http://localhost:9999/tts",
            auto_start=True,
        )
        config.gpt_sovits_root.mkdir(parents=True)
        (config.gpt_sovits_root / "api_v2.py").write_text("", encoding="utf-8")
        config.gpt_sovits_python.write_bytes(b"stub")
        config.gpt_sovits_config.write_text("stub", encoding="utf-8")
        engine = TTSEngine(config)

        with patch.object(
            engine, "_gpt_sovits_api_reachable", side_effect=(False, True)
        ), patch.object(subprocess, "Popen") as popen_mock, patch.object(
            tts_engine, "PROJECT_ROOT", self.test_root
        ):
            engine._ensure_gpt_sovits_api()

        command = popen_mock.call_args.args[0]
        self.assertEqual(command[command.index("-a") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("-p") + 1], "9999")


if __name__ == "__main__":
    unittest.main()
