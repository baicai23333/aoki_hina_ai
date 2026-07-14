import unittest

from response_text_policy import has_hidden_or_redacted_content


class ResponseTextPolicyTests(unittest.TestCase):
    def test_detects_markup_that_can_hide_source_content(self):
        cases = (
            "前文~~隐藏内容~~后文",
            "前文<!-- hidden -->后文",
            "前文<del>隐藏内容</del>后文",
            "前文<s>隐藏内容</s>后文",
            "前文<strike>隐藏内容</strike>后文",
            "<span hidden>隐藏内容</span>",
            '<span style="display:none">隐藏内容</span>',
            '<span style="visibility: hidden">隐藏内容</span>',
            '<span style="text-decoration: line-through">隐藏内容</span>',
            "删̶除̶内̶容̶",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(has_hidden_or_redacted_content(text))

    def test_detects_redaction_markers(self):
        for text in (
            "[已删除]",
            "【内容已屏蔽】",
            "[REDACTED]",
            "[削除済み]",
            "███",
        ):
            with self.subTest(text=text):
                self.assertTrue(has_hidden_or_redacted_content(text))

    def test_allows_ordinary_markdown_and_plain_text(self):
        for text in (
            "普通中文回复。",
            "**加粗内容**和《作品名》。",
            "- 第一项\n- 第二项",
            None,
        ):
            with self.subTest(text=text):
                self.assertFalse(has_hidden_or_redacted_content(text))


if __name__ == "__main__":
    unittest.main()
