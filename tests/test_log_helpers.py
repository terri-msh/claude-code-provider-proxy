import logging
import sys

sys.path.insert(0, "src")

from main import (
    LogEvent,
    LogRecord,
    PrettyConsoleFormatter,
    extract_last_user_prompt,
    mask_secrets,
)


# ---------------------------------------------------------------------------
# mask_secrets
# ---------------------------------------------------------------------------


class TestMaskSecrets:
    def test_long_authorization_header(self):
        headers = {
            "authorization": "OAuth y1__xCapOORpdT-ARiuKyCqndUCxxxxxxxxxxxxxxxxxxxxxxx",
            "content-type": "application/json",
        }
        masked = mask_secrets(headers)
        # First 8 chars of "OAuth y1__xCapOORpdT-..." are "OAuth y1"
        assert masked["authorization"] == "OAuth y1...XXX"
        assert masked["content-type"] == "application/json"

    def test_short_value(self):
        headers = {"x-api-key": "short"}
        masked = mask_secrets(headers)
        assert masked["x-api-key"] == "short...XXX"

    def test_no_secrets(self):
        headers = {"content-type": "application/json", "accept": "application/json"}
        masked = mask_secrets(headers)
        assert masked == headers

    def test_case_insensitive(self):
        headers = {"Authorization": "Bearer sk-1234567890abcdef"}
        masked = mask_secrets(headers)
        # First 8 chars of "Bearer sk-1234567890abcdef" are "Bearer s"
        assert masked["Authorization"] == "Bearer s...XXX"

    def test_x_api_key(self):
        headers = {"x-api-key": "sk-ant-api03-verylongkey"}
        masked = mask_secrets(headers)
        assert masked["x-api-key"] == "sk-ant-a...XXX"

    def test_api_key_header(self):
        headers = {"api-key": "1234567890abcdef"}
        masked = mask_secrets(headers)
        assert masked["api-key"] == "12345678...XXX"

    def test_cookie_header(self):
        headers = {"cookie": "session=abc123def456ghi789"}
        masked = mask_secrets(headers)
        # First 8 chars: "session="
        assert masked["cookie"] == "session=...XXX"

    def test_set_cookie_header(self):
        headers = {"set-cookie": "token=xyz987longvalue"}
        masked = mask_secrets(headers)
        assert masked["set-cookie"] == "token=xy...XXX"

    def test_proxy_authorization_header(self):
        headers = {"proxy-authorization": "Basic dXNlcjpwYXNz"}
        masked = mask_secrets(headers)
        assert masked["proxy-authorization"] == "Basic dX...XXX"

    def test_exactly_8_chars(self):
        """Values of exactly 8 chars are > 8 is False, so short path."""
        headers = {"authorization": "12345678"}
        masked = mask_secrets(headers)
        # len("12345678") == 8, which is NOT > 8, so short path
        assert masked["authorization"] == "12345678...XXX"

    def test_9_chars_long_path(self):
        """Values of 9 chars trigger the long path (> 8)."""
        headers = {"authorization": "123456789"}
        masked = mask_secrets(headers)
        assert masked["authorization"] == "12345678...XXX"

    def test_empty_value(self):
        headers = {"authorization": ""}
        masked = mask_secrets(headers)
        # len("") == 0, not > 8, so short path
        assert masked["authorization"] == "...XXX"

    def test_preserves_key_case(self):
        """Original key casing should be preserved."""
        headers = {"X-Api-Key": "longapikeyvalue123"}
        masked = mask_secrets(headers)
        assert "X-Api-Key" in masked
        # First 8 chars of "longapikeyvalue123": "longapik"
        assert masked["X-Api-Key"] == "longapik...XXX"


# ---------------------------------------------------------------------------
# extract_last_user_prompt
# ---------------------------------------------------------------------------


class TestExtractLastUserPrompt:
    def test_simple_string_content(self):
        body = {"messages": [{"role": "user", "content": "Hello, world!"}]}
        result = extract_last_user_prompt(body)
        assert result == "Hello, world!"

    def test_list_content_with_text_type(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Part one"},
                        {"type": "text", "text": "Part two"},
                    ],
                }
            ]
        }
        result = extract_last_user_prompt(body)
        assert result == "Part one Part two"

    def test_last_user_message_when_multiple(self):
        body = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ]
        }
        result = extract_last_user_prompt(body)
        assert result == "Second question"

    def test_truncation(self):
        long_text = "A" * 100
        body = {"messages": [{"role": "user", "content": long_text}]}
        result = extract_last_user_prompt(body, max_len=80)
        # max_len=80, so first 77 chars + "..."
        assert result == "A" * 77 + "..."
        assert len(result) == 80

    def test_custom_max_len(self):
        body = {"messages": [{"role": "user", "content": "Hello, world!"}]}
        result = extract_last_user_prompt(body, max_len=5)
        # max_len=5, so first 2 chars + "..."
        assert result == "He..."

    def test_no_messages(self):
        body = {}
        result = extract_last_user_prompt(body)
        assert result == ""

    def test_empty_messages(self):
        body = {"messages": []}
        result = extract_last_user_prompt(body)
        assert result == ""

    def test_no_user_messages(self):
        body = {"messages": [{"role": "assistant", "content": "No user here"}]}
        result = extract_last_user_prompt(body)
        assert result == ""

    def test_content_list_with_non_text_types(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": "data"},
                        {"type": "text", "text": "Only this text"},
                    ],
                }
            ]
        }
        result = extract_last_user_prompt(body)
        assert result == "Only this text"

    def test_content_list_with_only_non_text(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": "data"},
                    ],
                }
            ]
        }
        result = extract_last_user_prompt(body)
        assert result == ""

    def test_exact_max_len_no_truncation(self):
        text = "A" * 80
        body = {"messages": [{"role": "user", "content": text}]}
        result = extract_last_user_prompt(body, max_len=80)
        assert result == text
        assert "..." not in result

    def test_one_over_max_len_truncation(self):
        text = "A" * 81
        body = {"messages": [{"role": "user", "content": text}]}
        result = extract_last_user_prompt(body, max_len=80)
        assert result == "A" * 77 + "..."


# ---------------------------------------------------------------------------
# PrettyConsoleFormatter
# ---------------------------------------------------------------------------


def _make_log_record(event: str, message: str, request_id=None, data=None):
    """Helper to create a LogRecord (our custom dataclass)."""
    return LogRecord(
        event=event,
        message=message,
        request_id=request_id,
        data=data,
    )


def _make_python_log_record(log_record: LogRecord, level=logging.DEBUG):
    """Create a stdlib logging.LogRecord with our custom LogRecord attached."""
    record = logging.LogRecord(
        name="AnthropicProxy",
        level=level,
        pathname="test",
        lineno=0,
        msg=log_record.message,
        args=(),
        exc_info=None,
    )
    record.log_record = log_record
    return record


class TestPrettyConsoleFormatterNonStructured:
    def test_plain_message(self):
        formatter = PrettyConsoleFormatter()
        py_record = logging.LogRecord(
            name="AnthropicProxy",
            level=logging.INFO,
            pathname="test",
            lineno=0,
            msg="Simple log message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(py_record)
        assert "Simple log message" in result
        assert "INF" in result

    def test_uvicorn_message(self):
        formatter = PrettyConsoleFormatter()
        py_record = logging.LogRecord(
            name="uvicorn",
            level=logging.INFO,
            pathname="test",
            lineno=0,
            msg="Application startup complete",
            args=(),
            exc_info=None,
        )
        result = formatter.format(py_record)
        assert "Application startup complete" in result


class TestPrettyConsoleFormatterFallback:
    def test_unknown_event_with_data(self):
        formatter = PrettyConsoleFormatter()
        lr = _make_log_record(
            event="unknown_event",
            message="Something happened",
            data={"key1": "val1", "key2": "val2"},
        )
        py_record = _make_python_log_record(lr)
        result = formatter.format(py_record)
        assert "Something happened" in result
