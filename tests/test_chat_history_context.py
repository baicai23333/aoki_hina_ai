import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from chat_history_context import build_model_history


class ChatHistoryContextTests(unittest.TestCase):
    def test_hidden_assistant_content_is_not_replayed_to_the_model(self):
        records = [
            SimpleNamespace(type="human", content="之前聊了什么？"),
            SimpleNamespace(type="ai", content="这里有~~不应继续传播的内容~~。"),
            SimpleNamespace(type="human", content="换个普通话题吧。"),
            SimpleNamespace(type="ai", content="那就聊聊音乐。"),
        ]

        messages = build_model_history(records)

        self.assertEqual(
            [type(message) for message in messages],
            [HumanMessage, HumanMessage, AIMessage],
        )
        combined = "\n".join(str(message.content) for message in messages)
        self.assertNotIn("不应继续传播的内容", combined)
        self.assertIn("那就聊聊音乐", combined)

    def test_invalid_rows_are_ignored(self):
        records = [
            SimpleNamespace(type="system", content="not allowed"),
            SimpleNamespace(type="ai", content="   "),
            SimpleNamespace(type="human", content=None),
        ]

        self.assertEqual(build_model_history(records), [])


if __name__ == "__main__":
    unittest.main()
