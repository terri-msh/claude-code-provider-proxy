"""
Single-file FastAPI application to proxy Anthropic API requests to an OpenAI-compatible API (e.g., OpenRouter).
Handles request/response conversion, streaming, and dynamic model selection.
"""

import dataclasses
import enum
import json
import logging
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from logging.config import dictConfig
from pathlib import Path
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, List,
                    Literal, Optional, Tuple, Union, cast)
import re

import fastapi
import openai
import tiktoken
import uvicorn
from dotenv import load_dotenv, set_key
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai.types.chat import (ChatCompletionMessageParam,
                               ChatCompletionToolParam)
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

load_dotenv()


import yaml
import setproctitle

setproctitle.setproctitle("claude-code-proxy")

class ConnectionConfig(BaseModel):
    base_url: str
    api_key: str
    target_model: str
    provider: Optional[List[str]] = None
    allow_fallbacks: bool = False

class MappingsConfig(BaseModel):
    big_model: str
    medium_model: str
    small_model: str

class ProxyConfig(BaseModel):
    proxy_api_key: Optional[str] = None
    mappings: MappingsConfig
    connections: Dict[str, ConnectionConfig]

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file="../../.env", extra="ignore")

    config_path: str = "config.yaml"
    referer_url: str = "http://localhost:8080/claude_proxy"

    app_name: str = "AnthropicProxy"
    app_version: str = "0.2.0"
    log_level: str = "DEBUG"
    host: str = "127.0.0.1"
    port: int = 8080
    reload: bool = True

settings = Settings()

_console = Console()
_error_console = Console(stderr=True, style="bold red")

proxy_config: ProxyConfig
clients_pool: Dict[str, openai.AsyncClient] = {}

import httpx

VERBOSE_LOGGING = "--verbose" in sys.argv
LOG_JSON = "--log-json" in sys.argv

import contextvars
_current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_current_request_cost: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar("request_cost", default=None)
_current_request_provider: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_provider", default=None)

_SSE_LINE_RE = re.compile(r"data: (\{.*\})$", re.MULTILINE)


def _parse_sse_chunk(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE data line into JSON dict. Returns None if not a data line."""
    match = _SSE_LINE_RE.match(line.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_openrouter_usage(chunk: Dict[str, Any]) -> Tuple[Optional[float], int, int]:
    """Extract cost, cache_create, cache_read from OpenRouter SSE chunk.

    Returns: (cost, cache_create_tokens, cache_read_tokens)
    """
    cost = None
    cache_create = 0
    cache_read = 0

    usage = chunk.get("usage")
    if isinstance(usage, dict):
        # Cost from usage object
        cost_val = usage.get("cost")
        if cost_val is not None:
            cost = float(cost_val)

        # Cache tokens from prompt_tokens_details
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens", 0) or 0
            cache_create = details.get("cache_write_tokens", 0) or 0

    return cost, cache_create, cache_read

def truncate_large_structures(obj):
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "tools" and isinstance(v, list) and len(v) > 0:
                new_obj[k] = f"[{len(v)} tools truncated...]"
            elif k == "messages" and isinstance(v, list) and len(v) > 2:
                new_obj[k] = [f"[{len(v) - 1} messages truncated...]"] + [truncate_large_structures(v[-1])]
            else:
                new_obj[k] = truncate_large_structures(v)
        return new_obj
    elif isinstance(obj, list):
        if len(obj) > 10:
            return [truncate_large_structures(v) for v in obj[:3]] + [f"... [{len(obj)-6} items truncated] ..."] + [truncate_large_structures(v) for v in obj[-3:]]
        return [truncate_large_structures(v) for v in obj]
    elif isinstance(obj, str):
        if len(obj) > 300:
            return f"... [truncated {len(obj)-200} chars] ...\n{obj[-200:]}"
        return obj
    return obj

def format_log_body(body_str: str) -> str:
    try:
        parsed = json.loads(body_str)
        if not VERBOSE_LOGGING:
            parsed = truncate_large_structures(parsed)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        if not VERBOSE_LOGGING and len(body_str) > 500:
            return f"{body_str[:200]}\n... [truncated] ...\n{body_str[-200:]}"
        return body_str

_SECRET_HEADER_KEYS = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "proxy-authorization"}

def extract_cache_control_paths(obj: Any, current_path: str = "") -> List[str]:
    """Recursively finds all paths to 'cache_control' keys."""
    paths = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{current_path}.{k}" if current_path else k
            if k == "cache_control":
                paths.append(current_path if current_path else "root")
            else:
                paths.extend(extract_cache_control_paths(v, new_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{current_path}[{i}]"
            paths.extend(extract_cache_control_paths(v, new_path))
    return paths

def mask_secrets(headers: Dict[str, str]) -> Dict[str, str]:
    """Masks sensitive header values, showing first 8 chars + ...XXX."""
    masked = {}
    for key, value in headers.items():
        if key.lower() in _SECRET_HEADER_KEYS and len(value) > 8:
            masked[key] = value[:8] + "...XXX"
        elif key.lower() in _SECRET_HEADER_KEYS:
            masked[key] = value + "...XXX"
        else:
            masked[key] = value
    return masked

def extract_last_user_prompt(body: Dict[str, Any], max_len: int = 80) -> str:
    """Extracts the last user message content from an OpenAI-style request body."""
    messages = body.get("messages", [])
    last_user_content = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                last_user_content = " ".join(texts)
            elif isinstance(content, str):
                last_user_content = content
            break
    if len(last_user_content) > max_len:
        return last_user_content[:max_len - 3] + "..."
    return last_user_content

class YandexAuth(httpx.Auth):
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = self.token
        yield request

async def log_request_hook(request: httpx.Request):
    body_str = request.read().decode("utf-8", errors="ignore") if request.stream else ""
    request_id = _current_request_id.get()

    data: Dict[str, Any] = {
        "method": request.method,
        "url": str(request.url),
    }

    if body_str:
        try:
            parsed_body = json.loads(body_str)
            data["target_model"] = parsed_body.get("model", "?")
            data["stream"] = parsed_body.get("stream", False)
            data["last_user_prompt"] = extract_last_user_prompt(parsed_body)
            cache_paths = extract_cache_control_paths(parsed_body)
            if cache_paths:
                data["cache_breakpoints"] = cache_paths
        except (json.JSONDecodeError, AttributeError):
            data["body_preview"] = body_str[:200] if body_str else ""

    if VERBOSE_LOGGING:
        try:
            parsed_body = json.loads(body_str) if body_str else {}
            data["headers"] = mask_secrets(dict(request.headers))
            data["body"] = truncate_large_structures(parsed_body) if parsed_body else body_str
        except (json.JSONDecodeError, AttributeError):
            data["headers"] = mask_secrets(dict(request.headers))
            data["body"] = body_str

    debug(
        LogRecord(
            event=LogEvent.UPSTREAM_REQUEST.value,
            message=f"PROXY→Provider {request.method} {data.get('target_model', '?')}",
            request_id=request_id,
            data=data,
        )
    )

async def log_response_hook(response: httpx.Response):
    request_id = _current_request_id.get() if response.request else None
    content_type = response.headers.get("content-type", "")

    data: Dict[str, Any] = {
        "status_code": response.status_code,
        "url": str(response.url),
    }

    # Try multiple provider header variants
    provider_header = (
        response.headers.get("x-provider-name")
        or response.headers.get("x-provider")
    )
    if provider_header:
        data["provider"] = provider_header
        _current_request_provider.set(provider_header)

    # Try multiple cost header variants (OpenRouter uses different headers)
    cost_header = (
        response.headers.get("x-ratelimit-cost")
        or response.headers.get("x-cost")
        or response.headers.get("cost")
    )
    if cost_header:
        try:
            cost_val = float(cost_header)
            data["cost"] = cost_val
            _current_request_cost.set(cost_val)
        except (ValueError, TypeError):
            pass
    # Debug: log all response headers to discover cost/cache headers
    if VERBOSE_LOGGING:
        debug(LogRecord(
            event=LogEvent.UPSTREAM_RESPONSE.value,
            message="Response headers dump",
            request_id=request_id,
            data={"all_headers": mask_secrets(dict(response.headers))},
        ))

    is_sse = "text/event-stream" in content_type
    if is_sse:
        data["body_type"] = "sse_stream"
    else:
        try:
            await response.aread()
            body_text = response.text
            data["body_type"] = "json"
            try:
                parsed = json.loads(body_text)
                usage = parsed.get("usage", {})
                data["input_tokens"] = usage.get("prompt_tokens", 0)
                data["output_tokens"] = usage.get("completion_tokens", 0)
                # Extract provider from response body
                provider_val = parsed.get("provider") or parsed.get("providerName")
                if provider_val:
                    data["provider"] = provider_val
                    _current_request_provider.set(provider_val)
                # Extract cost from response body for non-streaming
                response_cost = usage.get('cost')
                if response_cost is not None:
                    data["cost"] = float(response_cost)
                    _current_request_cost.set(float(response_cost))
            except (json.JSONDecodeError, AttributeError):
                pass
            data["body_preview"] = body_text[:500] if body_text else ""
        except Exception:
            data["body_type"] = "unreadable"

    if VERBOSE_LOGGING:
        data["headers"] = mask_secrets(dict(response.headers))
        if not is_sse:
            try:
                data["body"] = format_log_body(body_text)
            except Exception:
                pass

    debug(
        LogRecord(
            event=LogEvent.UPSTREAM_RESPONSE.value,
            message=f"Provider→PROXY {response.status_code}",
            request_id=request_id,
            data=data,
        )
    )

_config_last_mtime: float = 0.0

def load_proxy_config(is_reload: bool = False) -> None:
    global proxy_config, clients_pool, _config_last_mtime
    try:
        mtime = os.path.getmtime(settings.config_path)
        with open(settings.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        new_proxy_config = ProxyConfig(**data)
    except Exception as e:
        if not is_reload:
            _error_console.print(f"[bold red]Configuration Error:[/bold red] Failed to load {settings.config_path}: {e}")
            sys.exit(1)
        else:
            error(LogRecord(event=LogEvent.MODEL_SELECTION.value, message=f"Failed to reload config: {e}"))
            return

    new_clients_pool = {}
    for conn_id, conn_cfg in new_proxy_config.connections.items():
        http_client = httpx.AsyncClient(
            event_hooks={'request': [log_request_hook], 'response': [log_response_hook]},
            verify=False,
            timeout=180.0,
            auth=YandexAuth(conn_cfg.api_key)
        )
        new_clients_pool[conn_id] = openai.AsyncClient(
            api_key=conn_cfg.api_key,
            base_url=conn_cfg.base_url,
            default_headers={
                "HTTP-Referer": settings.referer_url,
                "X-Title": settings.app_name,
            },
            http_client=http_client,
        )

    proxy_config = new_proxy_config
    clients_pool = new_clients_pool
    _config_last_mtime = mtime
    if is_reload:
        info(LogRecord(event=LogEvent.MODEL_SELECTION.value, message="Configuration reloaded successfully"))

def check_and_reload_config() -> None:
    try:
        current_mtime = os.path.getmtime(settings.config_path)
        if current_mtime > _config_last_mtime:
            load_proxy_config(is_reload=True)
    except OSError:
        pass

load_proxy_config()


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        header = {
            "timestamp": datetime.fromtimestamp(
                record.created, timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        log_payload = getattr(record, "log_record", None)
        if isinstance(log_payload, LogRecord):
            header["detail"] = dataclasses.asdict(log_payload)
        else:
            header["message"] = record.getMessage()
            if record.exc_info:
                exc_type, exc_value, exc_tb = record.exc_info
                header["error"] = {
                    "name": exc_type.__name__ if exc_type else "UnknownError",
                    "message": str(exc_value),
                    "stack_trace": "".join(
                        traceback.format_exception(exc_type, exc_value, exc_tb)
                    ),
                    "args": exc_value.args if hasattr(exc_value, "args") else [],
                }
        return json.dumps(header, ensure_ascii=False)


_LEVEL_STYLES = {
    "DEBUG":    ("dim cyan",    "DBG"),
    "INFO":     ("bold green",  "INF"),
    "WARNING":  ("bold yellow", "WRN"),
    "ERROR":    ("bold red",    "ERR"),
    "CRITICAL":  ("bold white on red", "CRT"),
}

_log_console = Console(highlight=False)


class PrettyConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        style, badge = _LEVEL_STYLES.get(record.levelname, ("default", record.levelname[:3]))

        log_payload: Optional["LogRecord"] = getattr(record, "log_record", None)

        if log_payload and isinstance(log_payload, LogRecord):
            return self._format_structured(ts, style, badge, log_payload)

        msg = record.getMessage()
        # uvicorn startup messages — keep short
        if record.name.startswith("uvicorn"):
            return f"[dim]{ts}[/] [{style}]{badge}[/] {msg}"
        return f"[dim]{ts}[/] [{style}]{badge}[/] {msg}"

    def _format_structured(self, ts: str, style: str, badge: str, rec: "LogRecord") -> str:
        event = rec.event
        data = rec.data or {}
        rid = f"[dim]#{rec.request_id[:8]}[/] " if rec.request_id else ""

        if event == LogEvent.REQUEST_START.value:
            client_m = data.get("client_model", "?")
            target_m = data.get("target_model", "?")
            stream = "⚡stream" if data.get("stream") else "sync"
            tokens = data.get("estimated_input_tokens", "?")
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]{client_m}[/] → [bold magenta]{target_m}[/] "
                f"[dim]({stream}, ~{tokens} tok)[/]"
            )

        if event == LogEvent.REQUEST_COMPLETED.value:
            dur = data.get("duration_ms", 0)
            inp = data.get("input_tokens", 0)
            out = data.get("output_tokens", 0)
            stop = data.get("stop_reason", "")
            dur_color = "green" if dur < 5000 else "yellow" if dur < 15000 else "red"
            cost = data.get("cost")
            cost_str = f" [green]${cost:.4f}[/]" if cost else ""
            provider = data.get("provider")
            provider_str = f" [blue]prov={provider}[/]" if provider else ""
            cache_create = data.get("cache_creation_input_tokens", 0)
            cache_read = data.get("cache_read_input_tokens", 0)
            cache_str = ""
            if cache_create or cache_read:
                cache_str = f" [dim]cache_create={cache_create} cache_read={cache_read}[/]"
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[{dur_color}]{dur:.0f}ms[/] "
                f"[dim]in={inp} out={out} stop={stop}[/]{cost_str}{provider_str}{cache_str}"
            )

        if event == LogEvent.REQUEST_FAILURE.value:
            err_type = data.get("error_type", "unknown")
            dur = data.get("duration_ms", 0)
            client_m = data.get("client_model", "")
            from rich.markup import escape
            model_info = f" \\[[bold]{escape(client_m)}[/]\\]" if client_m and client_m != "unknown" else ""
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold red]FAIL{model_info}[/] {rec.message} "
                f"[dim]({err_type}, {dur:.0f}ms)[/]"
            )

        if event == LogEvent.STREAM_INTERRUPTED.value:
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold red]STREAM ERROR[/] {rec.message}"
            )

        if event == LogEvent.MODEL_SELECTION.value:
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[yellow]⚠ {rec.message}[/]"
            )

        if event == LogEvent.TOKEN_COUNT.value:
            count = data.get("token_count", "?")
            model = data.get("model", "?")
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[dim]tokens: {count} ({model})[/]"
            )

        if event == LogEvent.ANTHROPIC_REQUEST.value:
            client_m = data.get("client_model", "?")
            cache_bps = data.get("cache_breakpoints")
            cache_str = f" [cyan]cache_breakpoints={cache_bps}[/]" if cache_bps else ""
            base = (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]Claude→Proxy[/] {client_m}{cache_str}"
            )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base

        if event == LogEvent.UPSTREAM_REQUEST.value:
            target_m = data.get("target_model", "?")
            stream = "⚡stream" if data.get("stream") else "sync"
            prompt = data.get("last_user_prompt", "")
            prompt_display = f' [dim]"{prompt}"[/]' if prompt else ""
            cache_bps = data.get("cache_breakpoints")
            cache_str = f" [cyan]cache_breakpoints={cache_bps}[/]" if cache_bps else ""
            base = (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[bold cyan]PROXY→Provider[/] {target_m} "
                f"[dim]{stream}[/]{prompt_display}{cache_str}"
            )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base

        if event == LogEvent.UPSTREAM_RESPONSE.value:
            status = data.get("status_code", "?")
            body_type = data.get("body_type", "")
            cost = data.get("cost")
            cost_str = f" [green]${cost:.4f}[/]" if cost else ""
            provider = data.get("provider")
            provider_str = f" [blue]prov={provider}[/]" if provider else ""
            if body_type == "sse_stream":
                base = (
                    f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                    f"[bold magenta]Provider→PROXY[/] {status} "
                    f"[dim]<SSE stream>[/]{cost_str}{provider_str}"
                )
            else:
                stop = data.get("stop_reason", "")
                inp = data.get("input_tokens", 0)
                out = data.get("output_tokens", 0)
                base = (
                    f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                    f"[bold magenta]Provider→PROXY[/] {status} {stop} "
                    f"[dim]in={inp} out={out}[/]{cost_str}{provider_str}"
                )
            if VERBOSE_LOGGING and (data.get("headers") or data.get("body") is not None):
                return base + "\n" + self._format_verbose_panels(data)
            return base

        if rec.error:
            return (
                f"[dim]{ts}[/] [{style}]{badge}[/] {rid}"
                f"[red]{rec.error.name}: {rec.error.message}[/]"
            )

        # fallback: just show message
        extra = ""
        if data:
            brief_keys = [k for k in data if k not in ("body", "response", "params")]
            if brief_keys:
                parts = [f"{k}={data[k]}" for k in brief_keys[:3]]
                extra = f" [dim]({', '.join(parts)})[/]"
        return f"[dim]{ts}[/] [{style}]{badge}[/] {rid}{rec.message}{extra}"

    def _format_verbose_panels(self, data: Dict[str, Any]) -> str:
        """Build Rich panel markup for headers and body in verbose mode."""
        parts = []
        headers = data.get("headers")
        if headers:
            header_lines = "\n".join(f"  {k}: {v}" for k, v in headers.items())
            parts.append(
                f"[dim]  ┌─ Headers ─────────────────────────────[/]\n"
                f"{header_lines}\n"
                f"[dim]  └──────────────────────────────────────[/]"
            )
        body = data.get("body")
        if body is not None:
            if isinstance(body, (dict, list)):
                body_str = json.dumps(body, indent=2, ensure_ascii=False)
            else:
                body_str = str(body)
            if len(body_str) > 2000:
                body_str = body_str[:1800] + "\n... [truncated] ..."
            body_lines = "\n".join(f"  {line}" for line in body_str.split("\n"))
            parts.append(
                f"[dim]  ┌─ Body ───────────────────────────────[/]\n"
                f"{body_lines}\n"
                f"[dim]  └──────────────────────────────────────[/]"
            )
        return "\n".join(parts)


class _RichHandler(logging.Handler):
    """Emits pre-formatted Rich markup to the console."""
    def __init__(self, formatter: logging.Formatter):
        super().__init__()
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _log_console.print(msg)
            if record.exc_info and record.exc_info[0] is not None:
                import traceback
                traceback.print_exception(*record.exc_info)
        except Exception:
            self.handleError(record)


_pretty_handler = _RichHandler(PrettyConsoleFormatter())

dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {"()": JSONFormatter},
        },
        "handlers": {},
        "loggers": {
            "": {"handlers": [], "level": "WARNING"},
            settings.app_name: {
                "handlers": [],
                "level": settings.log_level.upper(),
                "propagate": False,
            },
            "uvicorn": {"handlers": [], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": [], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": [], "level": "INFO", "propagate": False},
        },
    }
)

for _ln in [settings.app_name, "uvicorn", "uvicorn.error", "uvicorn.access"]:
    logging.getLogger(_ln).addHandler(_pretty_handler)


class LogEvent(enum.Enum):
    MODEL_SELECTION = "model_selection"
    REQUEST_START = "request_start"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILURE = "request_failure"
    ANTHROPIC_REQUEST = "anthropic_body"
    OPENAI_REQUEST = "openai_request"
    OPENAI_RESPONSE = "openai_response"
    ANTHROPIC_RESPONSE = "anthropic_response"
    STREAMING_REQUEST = "streaming_request"
    STREAM_INTERRUPTED = "stream_interrupted"
    TOKEN_COUNT = "token_count"
    UPSTREAM_REQUEST = "upstream_request"
    UPSTREAM_RESPONSE = "upstream_response"
    TOKEN_ENCODER_LOAD_FAILED = "token_encoder_load_failed"
    SYSTEM_PROMPT_ADJUSTED = "system_prompt_adjusted"
    TOOL_INPUT_SERIALIZATION_FAILURE = "tool_input_serialization_failure"
    IMAGE_FORMAT_UNSUPPORTED = "image_format_unsupported"
    MESSAGE_FORMAT_NORMALIZED = "message_format_normalized"
    TOOL_RESULT_SERIALIZATION_FAILURE = "tool_result_serialization_failure"
    TOOL_RESULT_PROCESSING = "tool_result_processing"
    TOOL_CHOICE_UNSUPPORTED = "tool_choice_unsupported"
    TOOL_ARGS_TYPE_MISMATCH = "tool_args_type_mismatch"
    TOOL_ARGS_PARSE_FAILURE = "tool_args_parse_failure"
    TOOL_ARGS_UNEXPECTED = "tool_args_unexpected"
    TOOL_ID_PLACEHOLDER = "tool_id_placeholder"
    TOOL_ID_UPDATED = "tool_id_updated"
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    HEALTH_CHECK = "health_check"
    PROVIDER_ERROR_DETAILS = "provider_error_details"


@dataclasses.dataclass
class LogError:
    name: str
    message: str
    stack_trace: Optional[str] = None
    args: Optional[Tuple[Any, ...]] = None


@dataclasses.dataclass
class LogRecord:
    event: str
    message: str
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[LogError] = None


_logger = logging.getLogger(settings.app_name)

if LOG_JSON:
    try:
        file_handler = logging.FileHandler("log.jsonl", mode="a")
        file_handler.setFormatter(JSONFormatter())
        _logger.addHandler(file_handler)
    except Exception as e:
        _error_console.print(
            f"Failed to configure JSON log file: {e}"
        )


def _log(level: int, record: LogRecord, exc: Optional[Exception] = None) -> None:
    if exc:
        record.error = LogError(
            name=type(exc).__name__,
            message=str(exc),
            stack_trace="".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            args=exc.args if hasattr(exc, "args") else tuple(),
        )
        if not record.message and str(exc):
            record.message = str(exc)
        elif not record.message:
            record.message = "An unspecified error occurred"

    _logger.log(level=level, msg=record.message, extra={"log_record": record})


def debug(record: LogRecord):
    _log(logging.DEBUG, record)


def info(record: LogRecord):
    _log(logging.INFO, record)


def warning(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.WARNING, record, exc=exc)


def error(record: LogRecord, exc: Optional[Exception] = None):
    if exc:
        import traceback
        traceback.print_exc()
    _log(logging.ERROR, record, exc=exc)


def critical(record: LogRecord, exc: Optional[Exception] = None):
    _log(logging.CRITICAL, record, exc=exc)


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ContentBlockImageSource(BaseModel):
    type: str
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: ContentBlockImageSource
    cache_control: Optional[Dict[str, Any]] = None


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]
    cache_control: Optional[Dict[str, Any]] = None


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], List[Any]]
    is_error: Optional[bool] = None
    cache_control: Optional[Dict[str, Any]] = None


ContentBlock = Union[
    ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult
]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(..., alias="input_schema")
    cache_control: Optional[Dict[str, Any]] = None


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "tool"]
    name: Optional[str] = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[ToolChoice] = None

    @field_validator("top_k")
    def check_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            req_id = info.context.get("request_id") if info.context else None
            warning(
                LogRecord(
                    event=LogEvent.PARAMETER_UNSUPPORTED.value,
                    message="Parameter 'top_k' provided by client but is not directly supported by the OpenAI Chat Completions API and will be ignored.",
                    request_id=req_id,
                    data={"parameter": "top_k", "value": v},
                )
            )
        return v


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ProviderErrorMetadata(BaseModel):
    provider_name: str
    raw_error: Optional[Dict[str, Any]] = None


class AnthropicErrorType(str, enum.Enum):
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    PERMISSION = "permission_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED = "overloaded_error"
    REQUEST_TOO_LARGE = "request_too_large_error"


class AnthropicErrorDetail(BaseModel):
    type: AnthropicErrorType
    message: str
    provider: Optional[str] = None
    provider_message: Optional[str] = None
    provider_code: Optional[Union[str, int]] = None


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicErrorDetail


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[ContentBlock]
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


STATUS_CODE_ERROR_MAP: Dict[int, AnthropicErrorType] = {
    400: AnthropicErrorType.INVALID_REQUEST,
    401: AnthropicErrorType.AUTHENTICATION,
    403: AnthropicErrorType.PERMISSION,
    404: AnthropicErrorType.NOT_FOUND,
    413: AnthropicErrorType.REQUEST_TOO_LARGE,
    422: AnthropicErrorType.INVALID_REQUEST,
    429: AnthropicErrorType.RATE_LIMIT,
    500: AnthropicErrorType.API_ERROR,
    502: AnthropicErrorType.API_ERROR,
    503: AnthropicErrorType.OVERLOADED,
    504: AnthropicErrorType.API_ERROR,
}


def extract_provider_error_details(
    error_details_dict: Optional[Dict[str, Any]],
) -> Optional[ProviderErrorMetadata]:
    if not isinstance(error_details_dict, dict):
        return None
    metadata = error_details_dict.get("metadata")
    if not isinstance(metadata, dict):
        return None
    provider_name = metadata.get("provider_name")
    raw_error_str = metadata.get("raw")

    if not provider_name or not isinstance(provider_name, str):
        return None

    parsed_raw_error: Optional[Dict[str, Any]] = None
    if isinstance(raw_error_str, str):
        # Many providers (e.g. Io Net, SiliconFlow) return plain text strings instead of JSON.
        # We try to parse it as JSON, but if it fails, we gracefully treat it as a plain message.
        try:
            parsed_raw_error = json.loads(raw_error_str)
        except json.JSONDecodeError:
            parsed_raw_error = {"message": raw_error_str}
    elif isinstance(raw_error_str, dict):
        parsed_raw_error = raw_error_str

    return ProviderErrorMetadata(
        provider_name=provider_name, raw_error=parsed_raw_error
    )





_token_encoder_cache: Dict[str, tiktoken.Encoding] = {}


def get_token_encoder(
    model_name: str = "gpt-4", request_id: Optional[str] = None
) -> tiktoken.Encoding:
    """Gets a tiktoken encoder, caching it for performance."""

    cache_key = "gpt-4"
    if cache_key not in _token_encoder_cache:
        try:
            _token_encoder_cache[cache_key] = tiktoken.encoding_for_model(cache_key)
        except Exception:
            try:
                _token_encoder_cache[cache_key] = tiktoken.get_encoding("cl100k_base")
                warning(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message=f"Could not load tiktoken encoder for '{cache_key}', using 'cl100k_base'. Token counts may be approximate.",
                        request_id=request_id,
                        data={"model_tried": cache_key},
                    )
                )
            except Exception as e_cl:
                critical(
                    LogRecord(
                        event=LogEvent.TOKEN_ENCODER_LOAD_FAILED.value,
                        message="Failed to load any tiktoken encoder (gpt-4, cl100k_base). Token counting will be inaccurate.",
                        request_id=request_id,
                    ),
                    exc=e_cl,
                )

                class DummyEncoder:
                    def encode(self, text: str) -> List[int]:
                        return list(range(len(text)))

                _token_encoder_cache[cache_key] = DummyEncoder()
    return _token_encoder_cache[cache_key]


def count_tokens_for_anthropic_request(
    messages: List[Message],
    system: Optional[Union[str, List[SystemContent]]],
    model_name: str,
    tools: Optional[List[Tool]] = None,
    request_id: Optional[str] = None,
) -> int:
    enc = get_token_encoder(model_name, request_id)
    total_tokens = 0

    if isinstance(system, str):
        total_tokens += len(enc.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, SystemContent) and block.type == "text":
                total_tokens += len(enc.encode(block.text))

    for msg in messages:
        total_tokens += 4
        if msg.role:
            total_tokens += len(enc.encode(msg.role))

        if isinstance(msg.content, str):
            total_tokens += len(enc.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ContentBlockText):
                    total_tokens += len(enc.encode(block.text))
                elif isinstance(block, ContentBlockImage):
                    total_tokens += 768
                elif isinstance(block, ContentBlockToolUse):
                    total_tokens += len(enc.encode(block.name))
                    try:
                        input_str = json.dumps(block.input)
                        total_tokens += len(enc.encode(input_str))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool input for token counting.",
                                data={"tool_name": block.name},
                                request_id=request_id,
                            )
                        )
                elif isinstance(block, ContentBlockToolResult):
                    try:
                        content_str = ""
                        if isinstance(block.content, str):
                            content_str = block.content
                        elif isinstance(block.content, list):
                            for item in block.content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    content_str += item.get("text", "")
                                else:
                                    content_str += json.dumps(item)
                        else:
                            content_str = json.dumps(block.content)
                        total_tokens += len(enc.encode(content_str))
                    except Exception:
                        warning(
                            LogRecord(
                                event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                                message="Failed to serialize tool result for token counting.",
                                request_id=request_id,
                            )
                        )

    if tools:
        total_tokens += 2
        for tool in tools:
            total_tokens += len(enc.encode(tool.name))
            if tool.description:
                total_tokens += len(enc.encode(tool.description))
            try:
                schema_str = json.dumps(tool.input_schema)
                total_tokens += len(enc.encode(schema_str))
            except Exception:
                warning(
                    LogRecord(
                        event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                        message="Failed to serialize tool schema for token counting.",
                        data={"tool_name": tool.name},
                        request_id=request_id,
                    )
                )
    debug(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Estimated {total_tokens} input tokens for model {model_name}",
            data={"model": model_name, "token_count": total_tokens},
            request_id=request_id,
        )
    )
    return total_tokens


StopReasonType = Optional[
    Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "error"]
]


def _serialize_tool_result_content_for_openai(
    anthropic_tool_result_content: Union[str, List[Dict[str, Any]], List[Any]],
    request_id: Optional[str],
    log_context: Dict,
) -> str:
    """
    Serializes Anthropic tool result content (which can be complex) into a single string
    as expected by OpenAI for the 'content' field of a 'tool' role message.
    """
    if isinstance(anthropic_tool_result_content, str):
        result_str = anthropic_tool_result_content
    elif isinstance(anthropic_tool_result_content, list):
        processed_parts = []
        contains_non_text_block = False
        for item in anthropic_tool_result_content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                processed_parts.append(str(item["text"]))
            else:
                try:
                    processed_parts.append(json.dumps(item))
                    contains_non_text_block = True
                except TypeError:
                    processed_parts.append(
                        f"<unserializable_item type='{type(item).__name__}'>"
                    )
                    contains_non_text_block = True

        result_str = "\n".join(processed_parts)
        if contains_non_text_block:
            warning(
                LogRecord(
                    event=LogEvent.TOOL_RESULT_PROCESSING.value,
                    message="Tool result content list contained non-text or complex items; parts were JSON stringified.",
                    request_id=request_id,
                    data={**log_context, "result_str_preview": result_str[:100]},
                )
            )
    else:
        try:
            result_str = json.dumps(anthropic_tool_result_content)
        except TypeError as e:
            warning(
                LogRecord(
                    event=LogEvent.TOOL_RESULT_SERIALIZATION_FAILURE.value,
                    message=f"Failed to serialize tool result content to JSON: {e}. Returning error JSON.",
                    request_id=request_id,
                    data=log_context,
                )
            )
            result_str = json.dumps(
                {
                    "error": "Serialization failed",
                    "original_type": str(type(anthropic_tool_result_content)),
                }
            )

    # Prevent models like GLM/DeepSeek from perceiving empty tool outputs as empty user messages
    if not result_str.strip():
        return "Tool executed successfully with no output."
    return result_str


import re

def convert_anthropic_to_openai_messages(
    anthropic_messages: List[Message],
    anthropic_system: Optional[Union[str, List[SystemContent]]] = None,
    request_id: Optional[str] = None,
    strip_cache: bool = False,
) -> List[Dict[str, Any]]:
    openai_messages: List[Dict[str, Any]] = []

    if isinstance(anthropic_system, str):
        if anthropic_system:
            openai_messages.append({"role": "system", "content": anthropic_system})
    elif isinstance(anthropic_system, list):
        sys_parts = []
        for block in anthropic_system:
            if isinstance(block, SystemContent) and block.type == "text":
                part: Dict[str, Any] = {"type": "text", "text": block.text}
                if getattr(block, "cache_control", None) and not strip_cache:
                    part["cache_control"] = block.cache_control
                sys_parts.append(part)
        
        if len(sys_parts) < len(anthropic_system):
            warning(
                LogRecord(
                    event=LogEvent.SYSTEM_PROMPT_ADJUSTED.value,
                    message="Non-text content blocks in Anthropic system prompt were ignored.",
                    request_id=request_id,
                )
            )
            
        if sys_parts:
            has_cache = any("cache_control" in part for part in sys_parts)
            if has_cache:
                openai_messages.append({"role": "system", "content": sys_parts})
            else:
                system_text_content = "\n".join(part["text"] for part in sys_parts)
                openai_messages.append({"role": "system", "content": system_text_content})

    for i, msg in enumerate(anthropic_messages):
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            openai_parts_for_user_message = []
            assistant_tool_calls = []
            text_content_for_assistant = []

            if not content and role == "user":
                openai_messages.append({"role": "user", "content": ""})
                continue
            if not content and role == "assistant":
                openai_messages.append({"role": "assistant", "content": ""})
                continue

            for block_idx, block in enumerate(content):
                block_log_ctx = {
                    "anthropic_message_index": i,
                    "block_index": block_idx,
                    "block_type": block.type,
                }

                if isinstance(block, ContentBlockText):
                    if role == "user":
                        openai_part: Dict[str, Any] = {"type": "text", "text": block.text}
                        if getattr(block, "cache_control", None) and not strip_cache:
                            openai_part["cache_control"] = block.cache_control
                        openai_parts_for_user_message.append(openai_part)
                    elif role == "assistant":
                        text_content_for_assistant.append(block.text)

                elif isinstance(block, ContentBlockImage) and role == "user":
                    if block.source.type == "base64":
                        img_part: Dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{block.source.media_type};base64,{block.source.data}"
                            },
                        }
                        if getattr(block, "cache_control", None) and not strip_cache:
                            img_part["cache_control"] = block.cache_control
                        openai_parts_for_user_message.append(img_part)
                    else:
                        warning(
                            LogRecord(
                                event=LogEvent.IMAGE_FORMAT_UNSUPPORTED.value,
                                message=f"Image block with source type '{block.source.type}' (expected 'base64') ignored in user message {i}.",
                                request_id=request_id,
                                data=block_log_ctx,
                            )
                        )

                elif isinstance(block, ContentBlockToolUse) and role == "assistant":
                    try:
                        args_str = json.dumps(block.input)
                    except Exception as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_INPUT_SERIALIZATION_FAILURE.value,
                                message=f"Failed to serialize tool input for tool '{block.name}'. Using empty JSON.",
                                request_id=request_id,
                                data={
                                    **block_log_ctx,
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                },
                            ),
                            exc=e,
                        )
                        args_str = "{}"

                    assistant_tool_calls.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": args_str},
                        }
                    )

                elif isinstance(block, ContentBlockToolResult) and role == "user":
                    serialized_content = _serialize_tool_result_content_for_openai(
                        block.content, request_id, block_log_ctx
                    )
                    tool_msg: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": serialized_content,
                    }
                    if getattr(block, "cache_control", None) and not strip_cache:
                        tool_msg["cache_control"] = block.cache_control
                    openai_messages.append(tool_msg)

            if role == "user" and openai_parts_for_user_message:
                has_cache_control = any(
                    "cache_control" in part for part in openai_parts_for_user_message
                )
                is_multimodal = any(
                    part["type"] == "image_url"
                    for part in openai_parts_for_user_message
                )
                if has_cache_control or is_multimodal or len(openai_parts_for_user_message) > 1:
                    openai_messages.append(
                        {"role": "user", "content": openai_parts_for_user_message}
                    )
                elif (
                    len(openai_parts_for_user_message) == 1
                    and openai_parts_for_user_message[0]["type"] == "text"
                ):
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": openai_parts_for_user_message[0]["text"],
                        }
                    )
            if role == "assistant":
                assistant_text = "\n".join(filter(None, text_content_for_assistant))
                if assistant_text:
                    openai_messages.append(
                        {"role": "assistant", "content": assistant_text}
                    )

                if assistant_tool_calls:
                    if (
                        openai_messages
                        and openai_messages[-1]["role"] == "assistant"
                    ):
                        openai_messages[-1]["tool_calls"] = assistant_tool_calls
                    else:
                        openai_messages.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": assistant_tool_calls,
                            }
                        )

    final_openai_messages = []
    for msg_dict in openai_messages:
        final_openai_messages.append(msg_dict)

    # Fix Claude Code cache busting: Claude Code injects a dynamic billing header 
    # into the system prompt AND user messages
    # e.g., 'x-anthropic-billing-header: cc_version=2.1.112.387; cc_entrypoint=cli; cch=b4911;'
    # The 'cch' value changes every request, which completely breaks exact prefix matching for caching.
    if final_openai_messages:
        for msg_dict in final_openai_messages:
            content = msg_dict.get("content")
            if isinstance(content, str):
                msg_dict["content"] = re.sub(r"cch=[0-9a-fA-F]+;", "cch=static;", content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and "text" in part:
                        part["text"] = re.sub(r"cch=[0-9a-fA-F]+;", "cch=static;", str(part["text"]))

    # Dump the exact outgoing payload to OpenAI for debugging
    try:
        import copy
        import hashlib
        import datetime
        
        session_id = None
        first_user_text = ""
        
        for msg in final_openai_messages:
            content = msg.get("content", "")
            content_str = content if isinstance(content, str) else str(content)
            
            if not session_id:
                match = re.search(r"cc_session=([a-zA-Z0-9_-]+)", content_str)
                if match:
                    session_id = match.group(1)
            
            if msg.get("role") == "user" and not first_user_text:
                first_user_text = content_str
                
        if not session_id:
            if first_user_text:
                session_id = hashlib.md5(first_user_text.encode("utf-8", errors="ignore")).hexdigest()[:8]
            else:
                session_id = "default"
                
        debug_messages = copy.deepcopy(final_openai_messages)
        
        tool_map = {}
        for msg in debug_messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                if len(msg["content"]) > 500:
                    msg["content"] = msg["content"][:500] + "... [TRUNCATED FOR DEBUG DUMP]"
            
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function":
                        func = tc.get("function", {})
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                            tool_map[tc["id"]] = {"name": func.get("name"), "args": args}
                            
                            changed = False
                            for k, v in args.items():
                                if isinstance(v, str) and len(v) > 500:
                                    args[k] = f"<{k} truncated, length={len(v)}>"
                                    changed = True
                            if changed:
                                func["arguments"] = json.dumps(args, ensure_ascii=False)
                        except Exception:
                            tool_map[tc["id"]] = {"name": func.get("name"), "args": {}}
            
            elif msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                tinfo = tool_map.get(tid, {})
                tname = tinfo.get("name", "tool")
                
                content = msg.get("content")
                if isinstance(content, str) and len(content) > 500:
                    if tname in ("Read", "ViewData", "read_file"):
                        file_path = tinfo.get("args", {}).get("file_path", "file")
                        file_name = str(file_path).split("/")[-1]
                        msg["content"] = f"<read {file_name}>"
                    elif tname in ("Bash", "run_command"):
                        command = tinfo.get("args", {}).get("command", "command")
                        msg["content"] = f"<bash '{str(command)[:50]}...'>"
                    elif tname in ("Grep", "grep_search"):
                        msg["content"] = f"<{tname} output truncated>"
                    else:
                        msg["content"] = content[:500] + f"... [{tname} output truncated]"

        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        date_str = now.strftime("%d_%m")
        dump_filename = f"sandbox/messages_dump_{date_str}_{session_id}.json"
        
        dump_data = {
            "request_timestamp": now.isoformat(),
            "messages": debug_messages
        }
        
        with open(dump_filename, "w") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        warning(
            LogRecord(
                event=LogEvent.REQUEST_FAILURE.value,
                message=f"Failed to dump messages: {e}",
            )
        )

    return final_openai_messages

def convert_anthropic_tools_to_openai(
    anthropic_tools: Optional[List[Tool]],
) -> Optional[List[Dict[str, Any]]]:
    if not anthropic_tools:
        return None
    res = []
    for t in anthropic_tools:
        func = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            },
        }
        if getattr(t, "cache_control", None):
            func["cache_control"] = t.cache_control
        res.append(func)
    return res


def convert_anthropic_tool_choice_to_openai(
    anthropic_choice: Optional[ToolChoice],
    request_id: Optional[str] = None,
) -> Optional[Union[str, Dict[str, Any]]]:
    if not anthropic_choice:
        return None
    if anthropic_choice.type == "auto":
        return "auto"
    if anthropic_choice.type == "any":
        warning(
            LogRecord(
                event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
                message="Anthropic tool_choice type 'any' mapped to OpenAI 'auto'. Exact behavior might differ (OpenAI 'auto' allows no tool use).",
                request_id=request_id,
                data={"anthropic_tool_choice": anthropic_choice.model_dump()},
            )
        )
        return "auto"
    if anthropic_choice.type == "tool" and anthropic_choice.name:
        return {"type": "function", "function": {"name": anthropic_choice.name}}

    warning(
        LogRecord(
            event=LogEvent.TOOL_CHOICE_UNSUPPORTED.value,
            message=f"Unsupported Anthropic tool_choice: {anthropic_choice.model_dump()}. Defaulting to 'auto'.",
            request_id=request_id,
            data={"anthropic_tool_choice": anthropic_choice.model_dump()},
        )
    )
    return "auto"


def convert_openai_to_anthropic_response(
    openai_response: openai.types.chat.ChatCompletion,
    original_anthropic_model_name: str,
    request_id: Optional[str] = None,
) -> MessagesResponse:
    anthropic_content: List[ContentBlock] = []
    anthropic_stop_reason: StopReasonType = None

    stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: "end_turn",
    }

    if openai_response.choices:
        choice = openai_response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason

        anthropic_stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        reasoning = getattr(message, "reasoning_content", "") or ""
        if not reasoning and getattr(message, "model_extra", None):
            reasoning = message.model_extra.get("reasoning_content", "") or message.model_extra.get("reasoning", "") or ""
            
        content_str = ""
        if reasoning:
            content_str += f"```thinking\n{reasoning}\n```\n"
        if message.content:
            content_str += message.content

        if content_str:
            anthropic_content.append(
                ContentBlockText(type="text", text=content_str)
            )

        if message.tool_calls:
            for call in message.tool_calls:
                if call.type == "function":
                    tool_input_dict: Dict[str, Any] = {}
                    try:
                        parsed_input = json.loads(call.function.arguments)
                        if isinstance(parsed_input, dict):
                            tool_input_dict = parsed_input
                        else:
                            tool_input_dict = {"value": parsed_input}
                            warning(
                                LogRecord(
                                    event=LogEvent.TOOL_ARGS_TYPE_MISMATCH.value,
                                    message=f"OpenAI tool arguments for '{call.function.name}' parsed to non-dict type '{type(parsed_input).__name__}'. Wrapped in 'value'.",
                                    request_id=request_id,
                                    data={
                                        "tool_name": call.function.name,
                                        "tool_id": call.id,
                                    },
                                )
                            )
                    except json.JSONDecodeError as e:
                        error(
                            LogRecord(
                                event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                                message=f"Failed to parse JSON arguments for tool '{call.function.name}'. Storing raw string.",
                                request_id=request_id,
                                data={
                                    "tool_name": call.function.name,
                                    "tool_id": call.id,
                                    "raw_args": call.function.arguments,
                                },
                            ),
                            exc=e,
                        )
                        tool_input_dict = {
                            "error_parsing_arguments": call.function.arguments
                        }

                    anthropic_content.append(
                        ContentBlockToolUse(
                            type="tool_use",
                            id=call.id,
                            name=call.function.name,
                            input=tool_input_dict,
                        )
                    )
            if finish_reason == "tool_calls":
                anthropic_stop_reason = "tool_use"

    if not anthropic_content:
        anthropic_content.append(ContentBlockText(type="text", text=""))

    usage = openai_response.usage
    anthropic_usage = Usage(
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )

    response_id = (
        f"msg_{openai_response.id}"
        if openai_response.id
        else f"msg_{request_id}_completed"
    )

    return MessagesResponse(
        id=response_id,
        type="message",
        role="assistant",
        model=original_anthropic_model_name,
        content=anthropic_content,
        stop_reason=anthropic_stop_reason,
        usage=anthropic_usage,
    )


def _get_anthropic_error_details_from_exc(
    exc: Exception,
) -> Tuple[AnthropicErrorType, str, int, Optional[ProviderErrorMetadata]]:
    """Maps caught exceptions to Anthropic error type, message, status code, and provider details."""
    error_type = AnthropicErrorType.API_ERROR
    error_message = str(exc)
    status_code = 500
    provider_details: Optional[ProviderErrorMetadata] = None

    if isinstance(exc, openai.APIError):
        error_message = getattr(exc, "message", None) or str(exc)
        status_code = getattr(exc, "status_code", None) or 500
        error_type = STATUS_CODE_ERROR_MAP.get(
            status_code, AnthropicErrorType.API_ERROR
        )

        if hasattr(exc, "body") and isinstance(exc.body, dict):
            actual_error_details = exc.body.get("error", exc.body)
            provider_details = extract_provider_error_details(actual_error_details)
            if provider_details and provider_details.provider_name:
                if error_message == "Provider returned error":
                    error_message = f"{provider_details.provider_name} returned error"
                else:
                    error_message = f"[{provider_details.provider_name}] {error_message}"

    if isinstance(exc, openai.AuthenticationError):
        error_type = AnthropicErrorType.AUTHENTICATION
    elif isinstance(exc, openai.RateLimitError):
        error_type = AnthropicErrorType.RATE_LIMIT
    elif isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        error_type = AnthropicErrorType.INVALID_REQUEST
    elif isinstance(exc, openai.PermissionDeniedError):
        error_type = AnthropicErrorType.PERMISSION
    elif isinstance(exc, openai.NotFoundError):
        error_type = AnthropicErrorType.NOT_FOUND

    return error_type, error_message, status_code, provider_details


def _format_anthropic_error_sse_event(
    error_type: AnthropicErrorType,
    message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> str:
    """Formats an error into the Anthropic SSE 'error' event structure."""
    anthropic_err_detail = AnthropicErrorDetail(type=error_type, message=message)
    if provider_details:
        anthropic_err_detail.provider = provider_details.provider_name
        if provider_details.raw_error and isinstance(
            provider_details.raw_error.get("error"), dict
        ):
            prov_err_obj = provider_details.raw_error["error"]
            anthropic_err_detail.provider_message = prov_err_obj.get("message")
            anthropic_err_detail.provider_code = prov_err_obj.get("code")
        elif provider_details.raw_error and isinstance(
            provider_details.raw_error.get("message"), str
        ):
            anthropic_err_detail.provider_message = provider_details.raw_error.get(
                "message"
            )
            anthropic_err_detail.provider_code = provider_details.raw_error.get("code")

    error_response = AnthropicErrorResponse(error=anthropic_err_detail)
    return f"event: error\ndata: {error_response.model_dump_json()}\n\n"


async def handle_anthropic_streaming_response_from_openai_stream(
    openai_stream: openai.AsyncStream[openai.types.chat.ChatCompletionChunk],
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
) -> AsyncGenerator[str, None]:
    """
    Consumes an OpenAI stream and yields Anthropic-compatible SSE events.
    BUGFIX: Correctly handles content block indexing for mixed text/tool_use.
    """

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"

    next_anthropic_block_idx = 0
    text_block_anthropic_idx: Optional[int] = None
    current_open_anthropic_block_idx: Optional[int] = None
    
    is_reasoning = False
    has_started_reasoning = False

    openai_tool_idx_to_anthropic_block_idx: Dict[int, int] = {}

    tool_states: Dict[int, Dict[str, Any]] = {}

    sent_tool_block_starts: set[int] = set()
    closed_anthropic_blocks: set[int] = set()

    output_token_count = 0
    final_anthropic_stop_reason: StopReasonType = None
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    enc = get_token_encoder(original_anthropic_model_name, request_id)

    openai_to_anthropic_stop_reason_map: Dict[Optional[str], StopReasonType] = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = 200
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value

    try:
        # Try to get raw SSE stream for cost/cache extraction (OpenRouter)
        raw_sse_extracted = False
        try:
            # Access underlying httpx response if available
            if hasattr(openai_stream, '_response') and hasattr(openai_stream._response, 'content'):
                raw_content = openai_stream._response.content.decode('utf-8', errors='ignore')
                raw_sse_extracted = True
                if VERBOSE_LOGGING:
                    debug(LogRecord(
                        event=LogEvent.STREAMING_REQUEST.value,
                        message="Raw SSE snapshot check",
                        request_id=request_id,
                        data={"raw_sse_available": True, "sample": raw_content[:2000] if raw_content else ""},
                    ))
        except Exception as e:
            if VERBOSE_LOGGING:
                debug(LogRecord(
                    event=LogEvent.STREAMING_REQUEST.value,
                    message="Could not access raw SSE stream",
                    request_id=request_id,
                    data={"error": str(e)},
                ))

        message_start_event_data = {
            "type": "message_start",
            "message": {
                "id": anthropic_message_id,
                "type": "message",
                "role": "assistant",
                "model": original_anthropic_model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_event_data)}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for chunk in openai_stream:
            # Extract provider
            provider_val = getattr(chunk, "provider", None) or getattr(chunk, "model_extra", {}).get("provider")
            if provider_val:
                _current_request_provider.set(provider_val)

            if not chunk.choices:
                # Extract cost and cache tokens from message_delta usage (OpenRouter)
                if chunk.usage:
                    # Debug: inspect chunk.usage structure
                    if VERBOSE_LOGGING:
                        debug(LogRecord(
                            event=LogEvent.STREAMING_REQUEST.value,
                            message="Chunk usage debug",
                            request_id=request_id,
                            data={
                                "usage_type": type(chunk.usage).__name__,
                                "usage_dict": chunk.usage.model_dump() if hasattr(chunk.usage, 'model_dump') else str(chunk.usage),
                                "usage_attrs": {k: v for k, v in chunk.usage.__dict__.items() if not k.startswith('_')} if hasattr(chunk.usage, '__dict__') else None,
                            },
                        ))

                    # Extract cost from chunk.usage (OpenRouter)
                    if hasattr(chunk.usage, 'cost'):
                        cost_value = getattr(chunk.usage, 'cost', None)
                        if cost_value is not None:
                            _current_request_cost.set(float(cost_value))

                    # Extract cache tokens - OpenRouter uses prompt_tokens_details.cached_tokens and cache_write_tokens
                    # Also try Anthropic-style names for compatibility
                    cache_read = 0
                    cache_write = 0

                    # Try OpenRouter style first
                    if hasattr(chunk.usage, 'prompt_tokens_details'):
                        details = getattr(chunk.usage, 'prompt_tokens_details', None)
                        if isinstance(details, dict):
                            cache_read = details.get('cached_tokens', 0) or 0
                            cache_write = details.get('cache_write_tokens', 0) or 0
                        elif hasattr(details, 'cached_tokens') or hasattr(details, 'cache_write_tokens'):
                            cache_read = getattr(details, 'cached_tokens', 0) or 0
                            cache_write = getattr(details, 'cache_write_tokens', 0) or 0

                    # Fallback to Anthropic-style names (if OpenRouter proxy provider uses them)
                    if not cache_read and not cache_write:
                        cache_write = getattr(chunk.usage, 'cache_creation_input_tokens', 0) or 0
                        cache_read = getattr(chunk.usage, 'cache_read_input_tokens', 0) or 0

                    # Direct attribute access as fallback
                    if not cache_write and not cache_read and hasattr(chunk.usage, '__dict__'):
                        dump_dict = chunk.usage.__dict__
                        cache_write = dump_dict.get('cache_write_tokens', 0) or dump_dict.get('cache_creation_input_tokens', 0) or 0
                        cache_read = dump_dict.get('cached_tokens', 0) or dump_dict.get('cache_read_input_tokens', 0) or 0

                    cache_creation_input_tokens = cache_write
                    cache_read_input_tokens = cache_read
                continue

            delta = chunk.choices[0].delta
            openai_finish_reason = chunk.choices[0].finish_reason

            with open("sandbox/stream_dump.jsonl", "a") as f:
                f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")

            reasoning = getattr(delta, "reasoning_content", "") or ""
            if not reasoning and getattr(delta, "model_extra", None):
                reasoning = delta.model_extra.get("reasoning_content", "") or delta.model_extra.get("reasoning", "") or ""
            
            delta_content = delta.content or ""

            if reasoning or delta_content:
                text_to_yield = ""
                if reasoning:
                    if not has_started_reasoning:
                        text_to_yield += "```thinking\n"
                        has_started_reasoning = True
                        is_reasoning = True
                    text_to_yield += reasoning
                
                if delta_content:
                    if is_reasoning:
                        text_to_yield += "\n```\n"
                        is_reasoning = False
                    text_to_yield += delta_content

                output_token_count += len(enc.encode(text_to_yield))
                
                if text_block_anthropic_idx is None:
                    text_block_anthropic_idx = next_anthropic_block_idx
                    next_anthropic_block_idx += 1
                
                if current_open_anthropic_block_idx != text_block_anthropic_idx:
                    if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx not in closed_anthropic_blocks:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
                        closed_anthropic_blocks.add(current_open_anthropic_block_idx)
                    current_open_anthropic_block_idx = text_block_anthropic_idx
                    start_text_event = {
                        "type": "content_block_start",
                        "index": text_block_anthropic_idx,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(start_text_event)}\n\n"

                text_delta_event = {
                    "type": "content_block_delta",
                    "index": text_block_anthropic_idx,
                    "delta": {"type": "text_delta", "text": text_to_yield},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(text_delta_event)}\n\n"

            if delta.tool_calls:
                for tool_delta in delta.tool_calls:
                    openai_tc_idx = tool_delta.index

                    if openai_tc_idx not in openai_tool_idx_to_anthropic_block_idx:
                        current_anthropic_tool_block_idx = next_anthropic_block_idx
                        next_anthropic_block_idx += 1
                        openai_tool_idx_to_anthropic_block_idx[openai_tc_idx] = (
                            current_anthropic_tool_block_idx
                        )

                        tool_states[current_anthropic_tool_block_idx] = {
                            "id": tool_delta.id
                            or f"tool_ph_{request_id}_{current_anthropic_tool_block_idx}",
                            "name": "",
                            "arguments_buffer": "",
                        }
                        if not tool_delta.id:
                            warning(
                                LogRecord(
                                    LogEvent.TOOL_ID_PLACEHOLDER.value,
                                    f"Generated placeholder Tool ID for OpenAI tool index {openai_tc_idx} -> Anthropic block {current_anthropic_tool_block_idx}",
                                    request_id,
                                )
                            )
                    else:
                        current_anthropic_tool_block_idx = (
                            openai_tool_idx_to_anthropic_block_idx[openai_tc_idx]
                        )

                    tool_state = tool_states[current_anthropic_tool_block_idx]

                    if tool_delta.id and tool_state["id"].startswith("tool_ph_"):
                        debug(
                            LogRecord(
                                LogEvent.TOOL_ID_UPDATED.value,
                                f"Updated placeholder Tool ID for Anthropic block {current_anthropic_tool_block_idx} to {tool_delta.id}",
                                request_id,
                            )
                        )
                        tool_state["id"] = tool_delta.id

                    if tool_delta.function:
                        if tool_delta.function.name:
                            tool_state["name"] = tool_delta.function.name
                        if tool_delta.function.arguments:
                            tool_state["arguments_buffer"] += (
                                tool_delta.function.arguments
                            )
                            output_token_count += len(
                                enc.encode(tool_delta.function.arguments)
                            )

                    if (
                        current_anthropic_tool_block_idx not in sent_tool_block_starts
                        and tool_state["name"]
                    ):
                        if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx != current_anthropic_tool_block_idx and current_open_anthropic_block_idx not in closed_anthropic_blocks:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
                            closed_anthropic_blocks.add(current_open_anthropic_block_idx)
                        current_open_anthropic_block_idx = current_anthropic_tool_block_idx
                        start_tool_event = {
                            "type": "content_block_start",
                            "index": current_anthropic_tool_block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_state["id"],
                                "name": tool_state["name"],
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(start_tool_event)}\n\n"
                        sent_tool_block_starts.add(current_anthropic_tool_block_idx)

                    if (
                        tool_delta.function
                        and tool_delta.function.arguments
                        and current_anthropic_tool_block_idx in sent_tool_block_starts
                    ):
                        args_delta_event = {
                            "type": "content_block_delta",
                            "index": current_anthropic_tool_block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tool_delta.function.arguments,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(args_delta_event)}\n\n"

            if openai_finish_reason:
                final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(
                    openai_finish_reason, "end_turn"
                )
                if openai_finish_reason == "tool_calls":
                    final_anthropic_stop_reason = "tool_use"
                break

        if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx not in closed_anthropic_blocks:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
            closed_anthropic_blocks.add(current_open_anthropic_block_idx)

        for anthropic_tool_idx in sent_tool_block_starts:
            tool_state_to_finalize = tool_states.get(anthropic_tool_idx)
            if tool_state_to_finalize:
                try:
                    json.loads(tool_state_to_finalize["arguments_buffer"])
                except json.JSONDecodeError:
                    warning(
                        LogRecord(
                            event=LogEvent.TOOL_ARGS_PARSE_FAILURE.value,
                            message=f"Buffered arguments for tool '{tool_state_to_finalize.get('name')}' (Anthropic block {anthropic_tool_idx}) did not form valid JSON.",
                            request_id=request_id,
                            data={
                                "buffered_args": tool_state_to_finalize[
                                    "arguments_buffer"
                                ][:100]
                            },
                        )
                    )

        if sent_tool_block_starts:
            final_anthropic_stop_reason = "tool_use"
        elif final_anthropic_stop_reason is None:
            final_anthropic_stop_reason = "end_turn"

        message_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_anthropic_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_token_count},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        stream_status_code = 500
        stream_log_event = LogEvent.REQUEST_FAILURE.value
        error_type, error_msg_str, _, provider_err_details = (
            _get_anthropic_error_details_from_exc(e)
        )
        stream_final_message = f"Error during OpenAI stream conversion: {error_msg_str}"
        final_anthropic_stop_reason = "error"

        error(
            LogRecord(
                event=LogEvent.STREAM_INTERRUPTED.value,
                message=stream_final_message,
                request_id=request_id,
                data={
                    "error_type": error_type.value,
                    "provider_details": provider_err_details.model_dump()
                    if provider_err_details
                    else None,
                },
            ),
            exc=e,
        )
        yield _format_anthropic_error_sse_event(
            error_type, error_msg_str, provider_err_details
        )

    finally:
        duration_ms = (time.monotonic() - start_time_mono) * 1000
        log_data = {
            "status_code": stream_status_code,
            "duration_ms": duration_ms,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
            "stop_reason": final_anthropic_stop_reason,
        }
        cost = _current_request_cost.get()
        provider_val = _current_request_provider.get()
        if provider_val:
            log_data["provider"] = provider_val
        # Debug: show if cost was captured
        if VERBOSE_LOGGING and cost is None:
            debug(LogRecord(
                event=LogEvent.REQUEST_COMPLETED.value,
                message="Cost not available from context var",
                request_id=request_id,
                data={"note": "cost header might not be in response headers or logging hook didn't capture it"},
            ))
        if cost is not None:
            log_data["cost"] = cost
        if cache_creation_input_tokens or cache_read_input_tokens:
            log_data["cache_creation_input_tokens"] = cache_creation_input_tokens
            log_data["cache_read_input_tokens"] = cache_read_input_tokens
        if stream_log_event == LogEvent.REQUEST_COMPLETED.value:
            info(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
        else:
            error(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
                            )


async def handle_anthropic_streaming_from_raw_httpx(
    httpx_response: httpx.Response,
    httpx_client: httpx.AsyncClient,
    original_anthropic_model_name: str,
    estimated_input_tokens: int,
    request_id: str,
    start_time_mono: float,
) -> AsyncGenerator[str, None]:
    """
    Consumes raw httpx SSE stream and yields Anthropic-compatible SSE events.
    Extracts OpenRouter cost and cache tokens from raw chunks.
    """

    anthropic_message_id = f"msg_stream_{request_id}_{uuid.uuid4().hex[:8]}"
    next_anthropic_block_idx = 0
    text_block_anthropic_idx: Optional[int] = None
    current_open_anthropic_block_idx: Optional[int] = None
    closed_anthropic_blocks = set()
    openai_tool_idx_to_anthropic_block_idx: Dict[int, int] = {}
    tool_states: Dict[int, Dict[str, Any]] = {}
    output_token_count = 0
    final_anthropic_stop_reason: StopReasonType = None
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0
    
    is_reasoning = False
    has_started_reasoning = False

    enc = get_token_encoder(original_anthropic_model_name, request_id)

    openai_to_anthropic_stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "stop_sequence",
        None: None,
    }

    stream_status_code = httpx_response.status_code
    stream_final_message = "Streaming request completed successfully."
    stream_log_event = LogEvent.REQUEST_COMPLETED.value

    resp_data: Dict[str, Any] = {"status_code": stream_status_code, "body_type": "sse_stream"}
    if VERBOSE_LOGGING:
        resp_data["headers"] = mask_secrets(dict(httpx_response.headers))
    debug(LogRecord(
        event=LogEvent.UPSTREAM_RESPONSE.value,
        message=f"Provider→PROXY {stream_status_code} <SSE stream>",
        request_id=request_id,
        data=resp_data,
    ))

    try:
        msg_start = {
            "type": "message_start",
            "message": {
                "id": anthropic_message_id,
                "type": "message",
                "role": "assistant",
                "model": original_anthropic_model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": estimated_input_tokens, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        async for line in httpx_response.aiter_lines():
            if not line:
                continue

            chunk_data = _parse_sse_chunk(line)
            if not chunk_data:
                continue

            provider_val = chunk_data.get("provider") or chunk_data.get("providerName")
            if provider_val:
                _current_request_provider.set(provider_val)

            cost, cache_write, cache_read = _extract_openrouter_usage(chunk_data)
            if cost is not None:
                _current_request_cost.set(cost)
            if cache_write:
                cache_creation_input_tokens = cache_write
            if cache_read:
                cache_read_input_tokens = cache_read

            if chunk_data.get("choices"):
                choices = chunk_data["choices"]
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")
                
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""

                if reasoning or content:
                    text_to_yield = ""
                    if reasoning:
                        if not has_started_reasoning:
                            text_to_yield += "```thinking\n"
                            has_started_reasoning = True
                            is_reasoning = True
                        text_to_yield += reasoning
                    
                    if content:
                        if is_reasoning:
                            text_to_yield += "\n```\n"
                            is_reasoning = False
                        text_to_yield += content
                        
                    output_token_count += len(enc.encode(text_to_yield))
                    
                    if text_block_anthropic_idx is None:
                        text_block_anthropic_idx = next_anthropic_block_idx
                        next_anthropic_block_idx += 1
                        
                    if current_open_anthropic_block_idx != text_block_anthropic_idx:
                        if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx not in closed_anthropic_blocks:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
                            closed_anthropic_blocks.add(current_open_anthropic_block_idx)
                        current_open_anthropic_block_idx = text_block_anthropic_idx
                        start_text_event = {
                            "type": "content_block_start",
                            "index": text_block_anthropic_idx,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(start_text_event)}\n\n"

                    text_delta_event = {
                        "type": "content_block_delta",
                        "index": text_block_anthropic_idx,
                        "delta": {"type": "text_delta", "text": text_to_yield},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(text_delta_event)}\n\n"

                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_idx = tool_call.get("index", 0)
                        if tool_idx not in openai_tool_idx_to_anthropic_block_idx:
                            new_idx = next_anthropic_block_idx
                            next_anthropic_block_idx += 1
                            openai_tool_idx_to_anthropic_block_idx[tool_idx] = new_idx
                            
                            call_id = tool_call.get("id")
                            if not call_id:
                                call_id = f"toolu_{uuid.uuid4().hex[:8]}"
                            tool_states[tool_idx] = {"name": "", "args_str": "", "started": False, "id": call_id}
                        else:
                            call_id = tool_call.get("id")
                            if call_id and tool_states[tool_idx]["id"].startswith("toolu_"):
                                if not tool_states[tool_idx]["started"]:
                                    tool_states[tool_idx]["id"] = call_id
                            
                        anthropic_idx = openai_tool_idx_to_anthropic_block_idx[tool_idx]
                        tool_state = tool_states[tool_idx]
                        name = tool_call.get("function", {}).get("name")
                        args_str = tool_call.get("function", {}).get("arguments")

                        if name:
                            tool_state["name"] = name
                            
                        if not tool_state["started"] and tool_state["name"]:
                            if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx not in closed_anthropic_blocks:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
                                closed_anthropic_blocks.add(current_open_anthropic_block_idx)
                            current_open_anthropic_block_idx = anthropic_idx
                            
                            tool_start_event = {
                                "type": "content_block_start",
                                "index": anthropic_idx,
                                "content_block": {"type": "tool_use", "id": tool_state["id"], "name": tool_state["name"], "input": {}},
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(tool_start_event)}\n\n"
                            tool_state["started"] = True
                        if args_str:
                            args_delta_event = {
                                "type": "content_block_delta",
                                "index": anthropic_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_str,
                                },
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(args_delta_event)}\n\n"

                if finish_reason:
                    final_anthropic_stop_reason = openai_to_anthropic_stop_reason_map.get(finish_reason, "end_turn")

            if line.strip() == "[DONE]":
                break

        if current_open_anthropic_block_idx is not None and current_open_anthropic_block_idx not in closed_anthropic_blocks:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_open_anthropic_block_idx})}\n\n"
            closed_anthropic_blocks.add(current_open_anthropic_block_idx)

        uncached_input_tokens = max(0, estimated_input_tokens - cache_creation_input_tokens - cache_read_input_tokens)
        usage_data = {
            "input_tokens": uncached_input_tokens,
            "output_tokens": output_token_count,
        }
        if cache_creation_input_tokens > 0:
            usage_data["cache_creation_input_tokens"] = cache_creation_input_tokens
        if cache_read_input_tokens > 0:
            usage_data["cache_read_input_tokens"] = cache_read_input_tokens
        delta_event = {
            "type": "message_delta",
            "delta": {"stop_reason": final_anthropic_stop_reason},
            "usage": usage_data,
        }
        yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        error_type = "stream_processing_error"
        final_anthropic_stop_reason = "error"
        stream_final_message = f"Error during stream processing: {str(e)}"
        stream_log_event = LogEvent.STREAM_INTERRUPTED.value
        stream_status_code = httpx_response.status_code or 500

        error(
            LogRecord(
                event=LogEvent.STREAM_INTERRUPTED.value,
                message=stream_final_message,
                request_id=request_id,
                data={
                    "error_type": error_type,
                    "traceback": traceback_str,
                },
            ),
            exc=e,
        )
        yield _format_anthropic_error_sse_event(error_type, str(e), None)

    finally:
        duration_ms = (time.monotonic() - start_time_mono) * 1000
        log_data = {
            "status_code": stream_status_code,
            "duration_ms": duration_ms,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_token_count,
            "stop_reason": final_anthropic_stop_reason,
        }
        cost = _current_request_cost.get()
        provider_val = _current_request_provider.get()
        if provider_val:
            log_data["provider"] = provider_val
        if cost is not None:
            log_data["cost"] = cost
        if cache_creation_input_tokens or cache_read_input_tokens:
            log_data["cache_creation_input_tokens"] = cache_creation_input_tokens
            log_data["cache_read_input_tokens"] = cache_read_input_tokens

        if stream_log_event == LogEvent.REQUEST_COMPLETED.value:
            info(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )
        else:
            error(
                LogRecord(
                    event=stream_log_event,
                    message=stream_final_message,
                    request_id=request_id,
                    data=log_data,
                )
            )

        # Cleanup httpx resources
        await httpx_response.aclose()
        await httpx_client.aclose()

from contextlib import asynccontextmanager
from pathlib import Path

@asynccontextmanager
async def app_lifespan(app: fastapi.FastAPI):
    try:
        dump_dir = Path("sandbox")
        dump_dir.mkdir(parents=True, exist_ok=True)
        for p in dump_dir.glob("messages_dump*.json"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception as e:
        warning(LogRecord(event=LogEvent.HEALTH_CHECK.value, message=f"Failed to clear messages dumps: {e}"))
    yield

app = fastapi.FastAPI(
    title=settings.app_name,
    description="Routes Anthropic API requests to an OpenAI-compatible API, selecting models dynamically.",
    version=settings.app_version,
    docs_url=None,
    redoc_url=None,
    lifespan=app_lifespan,
)

# ──────────────────── Cache Diff Diagnostic ────────────────────
import hashlib as _hashlib

_DIFF_LOG_PATH = Path(__file__).parent.parent / "cache_diff.log"
_prev_messages: Dict[str, List[str]] = {}  # model -> list of json-serialized messages


def _run_cache_diff_diagnostic(
    openai_messages: List[Dict[str, Any]], model: str, request_id: str
) -> None:
    """Compare current openai_messages with previous request, log diffs."""
    try:
        current = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in openai_messages]
        hashes = [_hashlib.sha256(s.encode()).hexdigest()[:8] for s in current]
        debug(LogRecord(LogEvent.OPENAI_REQUEST.value, f"msg hashes: {' '.join(f'[{i}]:{h}' for i, h in enumerate(hashes))}", request_id))

        prev = _prev_messages.get(model)
        _prev_messages[model] = current

        if prev is None:
            info(LogRecord(LogEvent.OPENAI_REQUEST.value, f"CACHE-DIFF first request for {model}, storing baseline ({len(current)} msgs)", request_id))
            return

        lines: list[str] = []
        lines.append(f"\n{'='*80}")
        lines.append(f"REQUEST {request_id}  model={model}  msgs: {len(prev)} -> {len(current)}")
        lines.append(f"{'='*80}")

        changed_indices = []
        max_common = min(len(prev), len(current))

        for i in range(max_common):
            if prev[i] == current[i]:
                continue
            changed_indices.append(i)
            role = openai_messages[i].get("role", "?")
            prev_s, curr_s = prev[i], current[i]
            diverge_pos = next(
                (p for p in range(min(len(prev_s), len(curr_s))) if prev_s[p] != curr_s[p]),
                min(len(prev_s), len(curr_s)),
            )
            ctx = 80
            start = max(0, diverge_pos - ctx)

            lines.append(f"\n  >>> [{i}] role={role}  diverges at char {diverge_pos}/{max(len(prev_s), len(curr_s))}")
            lines.append(f"    PREV[{start}:{diverge_pos+ctx}]: ...{prev_s[start:diverge_pos+ctx]}...")
            lines.append(f"    CURR[{start}:{diverge_pos+ctx}]: ...{curr_s[start:diverge_pos+ctx]}...")

            # Narrow zone around divergence
            fs, fe = max(0, diverge_pos - 30), diverge_pos + 30
            lines.append(f"    DIFF (+-30 chars @ {diverge_pos}):")
            lines.append(f"      OLD: <<{prev_s[fs:min(len(prev_s), fe)]}>>")
            lines.append(f"      NEW: <<{curr_s[fs:min(len(curr_s), fe)]}>>")

        if len(current) > len(prev):
            lines.append(f"\n  +++ {len(current) - len(prev)} NEW messages appended (indices {len(prev)}..{len(current)-1})")
        elif len(current) < len(prev):
            lines.append(f"\n  --- {len(prev) - len(current)} messages REMOVED")

        if not changed_indices and len(current) == len(prev):
            lines.append("\n  OK all existing messages IDENTICAL")

        diff_text = "\n".join(lines)

        with open(_DIFF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(diff_text + "\n")

        if changed_indices:
            info(LogRecord(LogEvent.OPENAI_REQUEST.value, f"CACHE-DIFF msgs changed: {changed_indices} (see cache_diff.log)", request_id))
        else:
            debug(LogRecord(LogEvent.OPENAI_REQUEST.value, f"CACHE-DIFF {max_common} common msgs identical, +{len(current)-len(prev)} new", request_id))

    except Exception as e:
        debug(LogRecord(LogEvent.OPENAI_REQUEST.value, f"CACHE-DIFF error: {e}", request_id))


def select_connection(client_model_name: str, request_id: str) -> Tuple[ConnectionConfig, openai.AsyncClient]:
    """Selects the target connection config and client based on the client's request."""
    check_and_reload_config()
    client_model_lower = client_model_name.lower()
    
    if "opus" in client_model_lower:
        target_conn_id = proxy_config.mappings.big_model
    elif "sonnet" in client_model_lower:
        target_conn_id = proxy_config.mappings.medium_model
    elif "haiku" in client_model_lower:
        target_conn_id = proxy_config.mappings.small_model
    else:
        target_conn_id = proxy_config.mappings.small_model
        warning(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message=f"Unknown client model '{client_model_name}', defaulting to SMALL mapping '{target_conn_id}'.",
                request_id=request_id,
            )
        )
    
    if target_conn_id not in proxy_config.connections:
        error_msg = f"Target connection '{target_conn_id}' is not defined in config.yaml"
        critical(LogRecord(event=LogEvent.MODEL_SELECTION.value, message=error_msg))
        raise Exception(error_msg)
        
    return proxy_config.connections[target_conn_id], clients_pool[target_conn_id]


def _build_anthropic_error_response(
    error_type: AnthropicErrorType,
    message: str,
    status_code: int,
    provider_details: Optional[ProviderErrorMetadata] = None,
) -> JSONResponse:
    """Creates a JSONResponse with Anthropic-formatted error."""
    err_detail = AnthropicErrorDetail(type=error_type, message=message)
    if provider_details:
        err_detail.provider = provider_details.provider_name
        if provider_details.raw_error:
            if isinstance(provider_details.raw_error, dict):
                prov_err_obj = provider_details.raw_error.get("error")
                if isinstance(prov_err_obj, dict):
                    err_detail.provider_message = prov_err_obj.get("message")
                    err_detail.provider_code = prov_err_obj.get("code")
                elif isinstance(provider_details.raw_error.get("message"), str):
                    err_detail.provider_message = provider_details.raw_error.get(
                        "message"
                    )
                    err_detail.provider_code = provider_details.raw_error.get("code")

    error_resp_model = AnthropicErrorResponse(error=err_detail)
    return JSONResponse(
        status_code=status_code, content=error_resp_model.model_dump(exclude_unset=True)
    )


async def _log_and_return_error_response(
    request: Request,
    status_code: int,
    anthropic_error_type: AnthropicErrorType,
    error_message: str,
    provider_details: Optional[ProviderErrorMetadata] = None,
    caught_exception: Optional[Exception] = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    start_time_mono = getattr(request.state, "start_time_monotonic", time.monotonic())
    duration_ms = (time.monotonic() - start_time_mono) * 1000

    log_data = {
        "status_code": status_code,
        "duration_ms": duration_ms,
        "error_type": anthropic_error_type.value,
        "client_ip": request.client.host if request.client else "unknown",
        "client_model": getattr(request.state, "client_model", "unknown"),
    }
    if provider_details:
        log_data["provider_name"] = provider_details.provider_name
        log_data["provider_raw_error"] = provider_details.raw_error

    error(
        LogRecord(
            event=LogEvent.REQUEST_FAILURE.value,
            message=f"Request failed: {error_message}",
            request_id=request_id,
            data=log_data,
        ),
        exc=caught_exception,
    )
    return _build_anthropic_error_response(
        anthropic_error_type, error_message, status_code, provider_details
    )


@app.get("/v1/models", tags=["API"], status_code=200)
async def list_models(request: Request):
    """Dummy endpoint to satisfy Claude Code's model checks."""
    return JSONResponse(content={
        "type": "list",
        "data": [
            {"type": "model", "id": "claude-sonnet-4-6", "display_name": "Sonnet 4.6", "created_at": "2024-01-01T00:00:00Z"},
            {"type": "model", "id": "claude-opus-4-6", "display_name": "Opus 4.6", "created_at": "2024-01-01T00:00:00Z"},
            {"type": "model", "id": "claude-haiku-4-5-20251001", "display_name": "Haiku 4.5", "created_at": "2024-01-01T00:00:00Z"}
        ]
    })

@app.post("/v1/messages", response_model=None, tags=["API"], status_code=200)
async def create_message_proxy(
    request: Request,
) -> Union[JSONResponse, StreamingResponse]:
    """
    Main endpoint for Anthropic message completions, proxied to an OpenAI-compatible API.
    Handles request/response conversions, streaming, and dynamic model selection.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time_monotonic = time.monotonic()
    _current_request_id.set(request_id)
    _current_request_cost.set(None)

    try:
        raw_body = await request.body()
        raw_json = json.loads(raw_body)
        client_model = raw_json.get("model", "unknown") if isinstance(raw_json, dict) else "unknown"
        request.state.client_model = client_model

        anthropic_request_data: Dict[str, Any] = {
            "client_model": client_model,
        }

        # Always extract cache breakpoints from raw incoming body (before Pydantic strips anything)
        incoming_cache_paths = extract_cache_control_paths(raw_json) if isinstance(raw_json, dict) else []
        if incoming_cache_paths:
            anthropic_request_data["cache_breakpoints_incoming"] = incoming_cache_paths
            debug(
                LogRecord(
                    LogEvent.ANTHROPIC_REQUEST.value,
                    f"Incoming cache breakpoints from Claude Code: {len(incoming_cache_paths)} breakpoint(s)",
                    request_id,
                    {"cache_breakpoints": incoming_cache_paths, "count": len(incoming_cache_paths)},
                )
            )

        if VERBOSE_LOGGING:
            anthropic_request_data["method"] = request.method
            anthropic_request_data["url"] = str(request.url)
            anthropic_request_data["headers"] = mask_secrets(dict(request.headers))
            anthropic_request_data["body"] = raw_json

        debug(
            LogRecord(
                LogEvent.ANTHROPIC_REQUEST.value,
                "Claude→Proxy",
                request_id,
                anthropic_request_data,
            )
        )
        
        api_key = request.headers.get("x-api-key")
        if proxy_config.proxy_api_key and api_key != proxy_config.proxy_api_key:
            return await _log_and_return_error_response(
                request,
                401,
                AnthropicErrorType.AUTHENTICATION,
                "Invalid API key",
            )

        anthropic_request = MessagesRequest.model_validate(
            raw_json, context={"request_id": request_id}
        )
    except json.JSONDecodeError as e:
        return await _log_and_return_error_response(
            request,
            400,
            AnthropicErrorType.INVALID_REQUEST,
            "Invalid JSON body.",
            caught_exception=e,
        )
    except ValidationError as e:
        return await _log_and_return_error_response(
            request,
            422,
            AnthropicErrorType.INVALID_REQUEST,
            f"Invalid request body: {e.errors()}",
            caught_exception=e,
        )

    is_stream = anthropic_request.stream or False
    target_conn, target_client = select_connection(anthropic_request.model, request_id)
    target_model_name = target_conn.target_model

    estimated_input_tokens = count_tokens_for_anthropic_request(
        messages=anthropic_request.messages,
        system=anthropic_request.system,
        model_name=anthropic_request.model,
        tools=anthropic_request.tools,
        request_id=request_id,
    )

    info(
        LogRecord(
            event=LogEvent.REQUEST_START.value,
            message="Processing new message request",
            request_id=request_id,
            data={
                "client_model": anthropic_request.model,
                "target_model": target_model_name,
                "stream": is_stream,
                "estimated_input_tokens": estimated_input_tokens,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )
    )

    is_implicit_caching_model = not (
        target_model_name.startswith("anthropic/") or target_model_name.startswith("google/")
    )

    try:
        openai_messages = convert_anthropic_to_openai_messages(
            anthropic_request.messages, 
            anthropic_request.system, 
            request_id=request_id,
            strip_cache=is_implicit_caching_model
        )
        openai_tools = convert_anthropic_tools_to_openai(anthropic_request.tools)
        openai_tool_choice = convert_anthropic_tool_choice_to_openai(
            anthropic_request.tool_choice, request_id
        )
    except Exception as e:
        return await _log_and_return_error_response(
            request,
            500,
            AnthropicErrorType.API_ERROR,
            "Error during request conversion.",
            caught_exception=e,
        )

    openai_params: Dict[str, Any] = {
        "model": target_model_name,
        "messages": cast(List[ChatCompletionMessageParam], openai_messages),
        "max_tokens": anthropic_request.max_tokens,
        "stream": is_stream,
    }

    if is_implicit_caching_model and openai_messages:
        _run_cache_diff_diagnostic(openai_messages, target_model_name, request_id)

    if is_stream:
        openai_params["stream_options"] = {"include_usage": True}
    if anthropic_request.temperature is not None:
        openai_params["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        openai_params["top_p"] = anthropic_request.top_p
    if anthropic_request.stop_sequences:
        openai_params["stop"] = anthropic_request.stop_sequences
    if openai_tools:
        openai_params["tools"] = cast(
            Optional[List[ChatCompletionToolParam]], openai_tools
        )
    if openai_tool_choice:
        openai_params["tool_choice"] = openai_tool_choice
    if anthropic_request.metadata and anthropic_request.metadata.get("user_id"):
        user_id_str = str(anthropic_request.metadata.get("user_id"))
        # openrouter rejects messages with 'user' longer than 128 characters
        openai_params["user"] = user_id_str[:128]

    if target_conn.provider:
        openai_params["extra_body"] = {
            "provider": {
                "order": target_conn.provider,
                "allow_fallbacks": target_conn.allow_fallbacks
            }
        }

    if is_implicit_caching_model:
        debug(
            LogRecord(
                LogEvent.OPENAI_REQUEST.value,
                "Stripped cache_control markers during conversion for implicit prefix caching model",
                request_id,
            )
        )

    # Log cache breakpoints in outgoing request at INFO level for visibility
    outgoing_cache_paths = extract_cache_control_paths(openai_params)
    if outgoing_cache_paths:
        info(
            LogRecord(
                LogEvent.OPENAI_REQUEST.value,
                f"Cache breakpoints in outgoing request: {len(outgoing_cache_paths)} breakpoint(s)",
                request_id,
                {"cache_breakpoints": outgoing_cache_paths, "count": len(outgoing_cache_paths)},
            )
        )
    else:
        warning(
            LogRecord(
                LogEvent.OPENAI_REQUEST.value,
                "No cache breakpoints found in outgoing OpenAI request",
                request_id,
            )
        )

    debug(
        LogRecord(
            LogEvent.OPENAI_REQUEST.value,
            "Prepared OpenAI request parameters",
            request_id,
            {"params": openai_params},
        )
    )

    try:
        if is_stream:
            debug(
                LogRecord(
                    LogEvent.STREAMING_REQUEST.value,
                    "Initiating streaming request to OpenAI-compatible API",
                    request_id,
                )
            )
            # Use raw httpx streaming for OpenRouter to capture metadata
            if True:
                httpx_client = httpx.AsyncClient(
                    verify=False,
                    timeout=180.0,
                )

                # Build correct URL - base_url already includes /v1
                endpoint_url = target_conn.base_url.rstrip('/') + '/chat/completions'

                # api_key in config already contains auth scheme prefix ("OAuth ..." or "Bearer ...")
                api_key = target_conn.api_key
                if not api_key.startswith(("OAuth ", "Bearer ")):
                    api_key = f"Bearer {api_key}"
                headers = {
                    "Authorization": api_key,
                    "Content-Type": "application/json",
                    "HTTP-Referer": settings.referer_url,
                    "X-Title": settings.app_name,
                }

                # Build JSON body: merge extra_body (OpenAI SDK convention) into top-level
                raw_body = {**openai_params, "stream": True}
                extra_body = raw_body.pop("extra_body", None)
                if extra_body and isinstance(extra_body, dict):
                    raw_body.update(extra_body)

                # Use send(stream=True) for true SSE streaming (post() buffers the whole response)
                req = httpx_client.build_request("POST", endpoint_url, headers=headers, json=raw_body, timeout=180.0)

                log_data: Dict[str, Any] = {
                    "target_model": raw_body.get("model", "?"),
                    "stream": True,
                    "last_user_prompt": extract_last_user_prompt(raw_body),
                }
                if VERBOSE_LOGGING:
                    log_data["headers"] = mask_secrets(dict(req.headers))
                    log_data["body"] = truncate_large_structures(raw_body)
                debug(LogRecord(
                    event=LogEvent.UPSTREAM_REQUEST.value,
                    message=f"PROXY→Provider {raw_body.get('model', '?')}",
                    request_id=request_id,
                    data=log_data,
                ))

                httpx_response = await httpx_client.send(req, stream=True)

                if httpx_response.status_code != 200:
                    await httpx_response.aread()
                    error_body = httpx_response.text
                    debug(LogRecord(
                        event=LogEvent.UPSTREAM_RESPONSE.value,
                        message=f"Provider→PROXY {httpx_response.status_code} ERROR",
                        request_id=request_id,
                        data={
                            "status_code": httpx_response.status_code,
                            "body": error_body[:500],
                            "headers": mask_secrets(dict(httpx_response.headers)) if VERBOSE_LOGGING else None,
                        },
                    ))
                    await httpx_response.aclose()
                    await httpx_client.aclose()
                    
                    try:
                        err_json = json.loads(error_body)
                        actual_err = err_json.get("error", err_json)
                        provider_details = extract_provider_error_details(actual_err)
                        err_msg = actual_err.get("message", error_body) if isinstance(actual_err, dict) else error_body
                    except Exception:
                        provider_details = None
                        err_msg = error_body
                        
                    err_type = STATUS_CODE_ERROR_MAP.get(httpx_response.status_code, AnthropicErrorType.API_ERROR)
                    
                    return await _log_and_return_error_response(
                        request,
                        httpx_response.status_code,
                        err_type,
                        err_msg,
                        provider_details,
                        None
                    )

                return StreamingResponse(
                    handle_anthropic_streaming_from_raw_httpx(
                        httpx_response,
                        httpx_client,
                        anthropic_request.model,
                        estimated_input_tokens,
                        request_id,
                        request.state.start_time_monotonic,
                    ),
                    media_type="text/event-stream",
                )
        else:
            debug(
                LogRecord(
                    LogEvent.OPENAI_REQUEST.value,
                    "Sending non-streaming request to OpenAI-compatible API",
                    request_id,
                )
            )
            openai_response_obj = await target_client.chat.completions.create(
                **openai_params
            )

            debug(
                LogRecord(
                    LogEvent.OPENAI_RESPONSE.value,
                    "Received OpenAI response",
                    request_id,
                    {"response": openai_response_obj.model_dump()},
                )
            )

            anthropic_response_obj = convert_openai_to_anthropic_response(
                openai_response_obj, anthropic_request.model, request_id=request_id
            )
            duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
            non_stream_data = {
                "status_code": 200,
                "duration_ms": duration_ms,
                "input_tokens": anthropic_response_obj.usage.input_tokens,
                "output_tokens": anthropic_response_obj.usage.output_tokens,
                "stop_reason": anthropic_response_obj.stop_reason,
            }
            cost = _current_request_cost.get()
            if cost is not None:
                non_stream_data["cost"] = cost
            raw_usage = openai_response_obj.usage
            if raw_usage:
                # Try OpenRouter style first (prompt_tokens_details.cached_tokens, cache_write_tokens)
                cache_create = 0
                cache_read = 0
                if hasattr(raw_usage, 'prompt_tokens_details'):
                    details = getattr(raw_usage, 'prompt_tokens_details', None)
                    if isinstance(details, dict):
                        cache_read = details.get('cached_tokens', 0) or 0
                        cache_create = details.get('cache_write_tokens', 0) or 0
                    elif hasattr(details, 'cached_tokens') or hasattr(details, 'cache_write_tokens'):
                        cache_read = getattr(details, 'cached_tokens', 0) or 0
                        cache_create = getattr(details, 'cache_write_tokens', 0) or 0

                # Fallback to Anthropic-style names
                if not cache_create and not cache_read:
                    cache_create = getattr(raw_usage, 'cache_creation_input_tokens', 0) or 0
                    cache_read = getattr(raw_usage, 'cache_read_input_tokens', 0) or 0

                # Direct attribute access as fallback
                if not cache_create and not cache_read and hasattr(raw_usage, '__dict__'):
                    dump_dict = raw_usage.__dict__
                    cache_create = dump_dict.get('cache_write_tokens', 0) or dump_dict.get('cache_creation_input_tokens', 0) or 0
                    cache_read = dump_dict.get('cached_tokens', 0) or dump_dict.get('cache_read_input_tokens', 0) or 0

                if cache_create or cache_read:
                    non_stream_data["cache_creation_input_tokens"] = cache_create
                    non_stream_data["cache_read_input_tokens"] = cache_read
            info(
                LogRecord(
                    event=LogEvent.REQUEST_COMPLETED.value,
                    message="Non-streaming request completed successfully",
                    request_id=request_id,
                    data=non_stream_data,
                )
            )
            debug(
                LogRecord(
                    LogEvent.ANTHROPIC_RESPONSE.value,
                    "Prepared Anthropic response",
                    request_id,
                    {"response": anthropic_response_obj.model_dump(exclude_unset=True)},
                )
            )
            return JSONResponse(
                content=anthropic_response_obj.model_dump(exclude_unset=True)
            )

    except openai.APIError as e:
        err_type, err_msg, err_status, prov_details = (
            _get_anthropic_error_details_from_exc(e)
        )
        return await _log_and_return_error_response(
            request, err_status, err_type, err_msg, prov_details, e
        )
    except Exception as e:
        return await _log_and_return_error_response(
            request,
            500,
            AnthropicErrorType.API_ERROR,
            "An unexpected error occurred while processing the request.",
            caught_exception=e,
        )


@app.post(
    "/v1/messages/count_tokens", response_model=TokenCountResponse, tags=["Utility"]
)
async def count_tokens_endpoint(request: Request) -> TokenCountResponse:
    """Estimates token count for given Anthropic messages and system prompt."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time_mono = time.monotonic()

    api_key = request.headers.get("x-api-key")
    if proxy_config.proxy_api_key and api_key != proxy_config.proxy_api_key:
        raise fastapi.HTTPException(status_code=401, detail="Invalid API key")

    try:
        body = await request.json()
        count_request = TokenCountRequest.model_validate(body)
    except json.JSONDecodeError as e:
        raise fastapi.HTTPException(status_code=400, detail="Invalid JSON body.") from e
    except ValidationError as e:
        raise fastapi.HTTPException(
            status_code=422, detail=f"Invalid request body: {e.errors()}"
        ) from e

    token_count = count_tokens_for_anthropic_request(
        messages=count_request.messages,
        system=count_request.system,
        model_name=count_request.model,
        tools=count_request.tools,
        request_id=request_id,
    )
    duration_ms = (time.monotonic() - start_time_mono) * 1000
    info(
        LogRecord(
            event=LogEvent.TOKEN_COUNT.value,
            message=f"Counted {token_count} tokens",
            request_id=request_id,
            data={
                "duration_ms": duration_ms,
                "token_count": token_count,
                "model": count_request.model,
            },
        )
    )
    return TokenCountResponse(input_tokens=token_count)


@app.get("/", include_in_schema=False, tags=["Health"])
async def root_health_check() -> JSONResponse:
    """Basic health check and information endpoint."""
    debug(
        LogRecord(
            event=LogEvent.HEALTH_CHECK.value, message="Root health check accessed"
        )
    )
    return JSONResponse(
        {
            "proxy_name": settings.app_name,
            "version": settings.app_version,
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.exception_handler(openai.APIError)
async def openai_api_error_handler(request: Request, exc: openai.APIError):
    err_type, err_msg, err_status, prov_details = _get_anthropic_error_details_from_exc(
        exc
    )
    return await _log_and_return_error_response(
        request, err_status, err_type, err_msg, prov_details, exc
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
    return await _log_and_return_error_response(
        request,
        422,
        AnthropicErrorType.INVALID_REQUEST,
        f"Validation error: {exc.errors()}",
        caught_exception=exc,
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
    return await _log_and_return_error_response(
        request,
        400,
        AnthropicErrorType.INVALID_REQUEST,
        "Invalid JSON format.",
        caught_exception=exc,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return await _log_and_return_error_response(
        request,
        500,
        AnthropicErrorType.API_ERROR,
        "An unexpected internal server error occurred.",
        caught_exception=exc,
    )


@app.middleware("http")
async def logging_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if not hasattr(request.state, "request_id"):
        request.state.request_id = str(uuid.uuid4())
    if not hasattr(request.state, "start_time_monotonic"):
        request.state.start_time_monotonic = time.monotonic()

    response = await call_next(request)

    response.headers["X-Request-ID"] = request.state.request_id
    duration_ms = (time.monotonic() - request.state.start_time_monotonic) * 1000
    response.headers["X-Response-Time-ms"] = str(duration_ms)

    return response


def update_env_file() -> None:
    """Updates the .env file with current proxy settings for Claude Code."""
    try:
        env_path = Path(os.path.dirname(__file__)).parent.parent / ".env"
        proxy_url = f"http://{settings.host}:{settings.port}"
        proxy_api_key = proxy_config.proxy_api_key or "proxy-key"
        
        # Resolve to absolute path for clear logging
        abs_env_path = env_path.resolve()
        
        set_key(str(abs_env_path), "ANTHROPIC_BASE_URL", proxy_url)
        set_key(str(abs_env_path), "ANTHROPIC_API_KEY", proxy_api_key)
        
        _console.print(f"[bold green]Updated {abs_env_path} with proxy variables.[/bold green]")
    except Exception as e:
        _error_console.print(f"[bold red]Environment Error:[/bold red] Failed to update .env: {e}")

if __name__ == "__main__":
    update_env_file()
    _console.print(
        r"""[bold blue]
           /$$                           /$$
          | $$                          | $$
  /$$$$$$$| $$  /$$$$$$  /$$   /$$  /$$$$$$$  /$$$$$$         /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$ /$$   /$$
 /$$_____/| $$ |____  $$| $$  | $$ /$$__  $$ /$$__  $$       /$$__  $$ /$$__  $$ /$$__  $$|  $$ /$$/| $$  | $$
| $$      | $$  /$$$$$$$| $$  | $$| $$  | $$| $$$$$$$$      | $$  \ $$| $$  \__/| $$  \ $$ \  $$$$/ | $$  | $$
| $$      | $$ /$$__  $$| $$  | $$| $$  | $$| $$_____/      | $$  | $$| $$      | $$  | $$  >$$  $$ | $$  | $$
|  $$$$$$$| $$|  $$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$      | $$$$$$$/| $$      |  $$$$$$/ /$$/\  $$|  $$$$$$$
 \_______/|__/ \_______/ \______/  \_______/ \_______/      | $$____/ |__/       \______/ |__/  \__/ \____  $$
                                                            | $$                                     /$$  | $$
                                                            | $$                                    |  $$$$$$/
                                                            |__/                                     \______/ 
    [/]""",
        justify="left",
    )
    config_details_text = Text.assemble(
        ("   Version       : ", "default"),
        (f"v{settings.app_version}", "bold cyan"),
        ("\n   Big Model     : ", "default"),
        (proxy_config.mappings.big_model, "magenta"),
        ("\n   Med Model     : ", "default"),
        (proxy_config.mappings.medium_model, "cyan"),
        ("\n   Small Model   : ", "default"),
        (proxy_config.mappings.small_model, "green"),
        ("\n   Log Level     : ", "default"),
        (settings.log_level.upper(), "yellow"),
        ("\n   Log JSON      : ", "default"),
        ("Enabled", "bold green") if LOG_JSON else ("Disabled", "dim"),
        ("\n   Listening on  : ", "default"),
        (f"http://{settings.host}:{settings.port}", "bold white"),
        ("\n   Reload        : ", "default"),
        ("Enabled", "bold orange1") if settings.reload else ("Disabled", "dim"),
    )
    _console.print(
        Panel(
            config_details_text,
            title="Anthropic Proxy Configuration",
            border_style="blue",
            expand=False,
        )
    )
    _console.print(Rule("Starting Uvicorn server...", style="dim blue"))

    uvicorn.run(
        "__main__:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,
        access_log=True,
    )
