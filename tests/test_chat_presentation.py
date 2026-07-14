import unittest

from chat_presentation import translation_status_message


class TranslationStatusMessageTests(unittest.TestCase):
    def test_rejected_and_failed_use_fixed_public_messages(self):
        self.assertEqual(
            translation_status_message("rejected"),
            "日语译文未通过安全复核，本条仅显示中文，未生成语音。",
        )
        self.assertEqual(
            translation_status_message("failed"),
            "日语翻译暂时失败，本条仅显示中文，未生成语音。",
        )

    def test_other_or_untrusted_statuses_are_hidden(self):
        statuses = (
            "validated",
            "fixed",
            "none",
            "legacy_unverified",
            "raw exception: secret path",
            None,
            123,
        )
        for status in statuses:
            with self.subTest(status=status):
                self.assertIsNone(translation_status_message(status))


if __name__ == "__main__":
    unittest.main()
