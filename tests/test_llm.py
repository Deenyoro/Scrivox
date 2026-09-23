"""Unit tests for the LLM client and translation response parsing.

No network access: requests.post and the backoff sleep are patched.
Run with:  python -m unittest discover -s tests
"""

import threading
import unittest
from unittest import mock

from scrivox.core import llm_client
from scrivox.core.llm_client import (
    _convert_openai_to_anthropic_messages,
    _retry_delay,
    chat_completion,
    is_error_response,
)
from scrivox.core.translator import _parse_numbered_lines


class _Resp:
    def __init__(self, status, body=None, headers=None, text=""):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _openai_ok(text):
    return _Resp(200, {"choices": [{"message": {"content": text}}]})


class ParseNumberedLinesTests(unittest.TestCase):
    def test_numbers_inside_content_do_not_hijack_numbering(self):
        self.assertEqual(
            _parse_numbered_lines("1: Hola\n2: 12. de mayo\n3: a las 1:30", 3),
            ["Hola", "12. de mayo", "a las 1:30"],
        )

    def test_unnumbered_line_is_continuation(self):
        self.assertEqual(_parse_numbered_lines("1: Hola\ncontinua\n2: Adios", 2),
                         ["Hola continua", "Adios"])

    def test_trailing_commentary_is_not_glued_on(self):
        self.assertEqual(_parse_numbered_lines("1: a\n2: b\nNote: x", 2), ["a", "b"])

    def test_plain_lines_fallback_requires_exact_count(self):
        self.assertEqual(_parse_numbered_lines("uno\ndos", 2), ["uno", "dos"])
        self.assertIsNone(_parse_numbered_lines("uno\ndos\ntres", 2))
        self.assertIsNone(_parse_numbered_lines("", 1))


class MessageConversionTests(unittest.TestCase):
    def test_system_messages_extracted_and_images_converted(self):
        converted, system = _convert_openai_to_anthropic_messages([
            {"role": "system", "content": "A"},
            {"role": "system", "content": [{"type": "text", "text": "B"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                {"type": "image_url", "image_url": {"url": "https://example.invalid/x.jpg"}},
            ]},
        ])
        self.assertEqual(system, "A\n\nB")
        self.assertEqual(converted, [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "QUJD"}},
            {"type": "image", "source": {"type": "url",
                                         "url": "https://example.invalid/x.jpg"}},
        ]}])


class RetryDelayTests(unittest.TestCase):
    def test_retry_after_is_honored_and_clamped(self):
        self.assertEqual(_retry_delay(_Resp(429, headers={"Retry-After": "7"}), 0), 7.0)
        self.assertEqual(_retry_delay(_Resp(429, headers={"Retry-After": "999"}), 0), 60.0)
        self.assertEqual(_retry_delay(_Resp(429, headers={"Retry-After": "soon"}), 1), 10.0)
        self.assertEqual(_retry_delay(_Resp(503), 2), 4.0)


@mock.patch.object(llm_client, "_cancellable_sleep", return_value=False)
class ChatCompletionTests(unittest.TestCase):
    URL = "https://openrouter.example.invalid/api/v1/chat/completions"

    def test_retries_5xx_then_returns_text(self, _sleep):
        with mock.patch.object(llm_client.requests, "post",
                               side_effect=[_Resp(502), _openai_ok(" done ")]) as post:
            out = chat_completion([{"role": "user", "content": "x"}], "m", "k", self.URL)
        self.assertEqual(out, "done")
        self.assertEqual(post.call_count, 2)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 2000)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k")

    def test_429_is_retried_after_retry_after_delay(self, sleep):
        with mock.patch.object(llm_client.requests, "post",
                               side_effect=[_Resp(429, headers={"Retry-After": "3"}),
                                            _openai_ok("ok")]) as post:
            out = chat_completion([{"role": "user", "content": "x"}], "m", "k", self.URL)
        self.assertEqual(out, "ok")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(sleep.call_args.args[0], 3.0)

    def test_anthropic_client_error_is_reported(self, _sleep):
        with mock.patch.object(llm_client.requests, "post",
                               return_value=_Resp(400, text="bad request")) as post:
            out = chat_completion([{"role": "user", "content": "x"}], "claude-model",
                                  "sk-ant", "https://api.anthropic.com/v1/messages")
        self.assertEqual(post.call_count, 1)
        self.assertTrue(is_error_response(out))
        self.assertIn("400", out)

    def test_client_error_is_not_retried(self, _sleep):
        with mock.patch.object(llm_client.requests, "post",
                               return_value=_Resp(401, text="bad key")) as post:
            out = chat_completion([{"role": "user", "content": "x"}], "m", "k", self.URL)
        self.assertEqual(post.call_count, 1)
        self.assertTrue(is_error_response(out))
        self.assertIn("401", out)

    def test_empty_content_is_retried_then_reported(self, _sleep):
        with mock.patch.object(llm_client.requests, "post",
                               return_value=_openai_ok("   ")) as post:
            out = chat_completion([{"role": "user", "content": "x"}], "m", "k", self.URL,
                                  max_retries=2)
        self.assertEqual(post.call_count, 2)
        self.assertTrue(is_error_response(out))

    def test_cancel_before_request(self, _sleep):
        ev = threading.Event()
        ev.set()
        with mock.patch.object(llm_client.requests, "post") as post:
            out = chat_completion([{"role": "user", "content": "x"}], "m", "k", self.URL,
                                  cancel_event=ev)
        post.assert_not_called()
        self.assertEqual(out, "[Cancelled]")

    def test_anthropic_endpoint_uses_messages_format(self, _sleep):
        resp = _Resp(200, {"content": [{"type": "text", "text": "hi"}]})
        with mock.patch.object(llm_client.requests, "post", return_value=resp) as post:
            out = chat_completion(
                [{"role": "system", "content": "sys"}, {"role": "user", "content": "x"}],
                "claude-model", "sk-ant", "https://api.anthropic.com/v1/messages",
            )
        self.assertEqual(out, "hi")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["x-api-key"], "sk-ant")
        self.assertEqual(kwargs["json"]["system"], "sys")
        self.assertEqual(kwargs["json"]["messages"], [{"role": "user", "content": "x"}])

    def test_openai_endpoint_uses_max_completion_tokens(self, _sleep):
        with mock.patch.object(llm_client.requests, "post",
                               return_value=_openai_ok("ok")) as post:
            chat_completion([{"role": "user", "content": "x"}], "m", "k",
                            "https://api.openai.com/v1/chat/completions", max_tokens=5)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 5)
        self.assertNotIn("max_tokens", payload)


class ErrorSentinelTests(unittest.TestCase):
    def test_content_starting_with_bracket_is_not_an_error(self):
        self.assertFalse(is_error_response("[Music] intro"))
        self.assertTrue(is_error_response("[API error 500 after 3 retries]"))
        self.assertTrue(is_error_response(""))
        self.assertTrue(is_error_response(None))


if __name__ == "__main__":
    unittest.main()
