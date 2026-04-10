from __future__ import annotations

import os
from typing import Any


DEFAULT_OPENAI_SERVICE_TIER = "flex"
FALLBACK_OPENAI_SERVICE_TIER = "auto"
OPENAI_SERVICE_TIER_ENV = "OPENAI_SERVICE_TIER"
OPENAI_MAX_COMPLETION_TOKENS_PREFIXES = ("gpt-5",)


def resolve_openai_service_tier(provider: str, configured_tier: str | None = None) -> str | None:
    if provider.strip().lower() != "openai":
        return None
    candidate = configured_tier
    if candidate is None:
        candidate = os.environ.get(OPENAI_SERVICE_TIER_ENV, "")
    tier = candidate.strip().lower()
    return tier or DEFAULT_OPENAI_SERVICE_TIER


def create_chat_completion(client: Any, **request_kwargs: Any) -> Any:
    request = dict(request_kwargs)
    provider = getattr(client, "_provider", "").strip().lower()
    requested_tier = None
    if provider == "openai":
        requested_tier = getattr(client, "_openai_service_tier", DEFAULT_OPENAI_SERVICE_TIER)
        if requested_tier and "service_tier" not in request:
            request["service_tier"] = requested_tier
        request = _normalize_openai_request(request)

    seen_retry_keys: set[str] = set()
    while True:
        try:
            return client.chat.completions.create(**request)
        except Exception as exc:
            retry_key, retry_request, retry_message = _build_retry_request(
                exc,
                request=request,
                requested_tier=requested_tier,
            )
            if retry_request is None or retry_key in seen_retry_keys:
                raise
            seen_retry_keys.add(retry_key)
            print(retry_message)
            request = retry_request


def _normalize_openai_request(request: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(request)
    model = str(normalized.get("model", "")).strip().lower()
    if _openai_model_uses_max_completion_tokens(model) and "max_tokens" in normalized:
        normalized["max_completion_tokens"] = normalized.pop("max_tokens")
    return normalized


def _openai_model_uses_max_completion_tokens(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in OPENAI_MAX_COMPLETION_TOKENS_PREFIXES)


def _build_retry_request(
    exc: Exception,
    *,
    request: dict[str, Any],
    requested_tier: str | None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    if _should_retry_with_max_completion_tokens(exc, request):
        retry_request = dict(request)
        retry_request["max_completion_tokens"] = retry_request.pop("max_tokens")
        return (
            "max_completion_tokens",
            retry_request,
            "OpenAI model rejected max_tokens; retrying with max_completion_tokens.",
        )
    if _should_retry_with_standard_processing(exc, requested_tier):
        retry_request = dict(request)
        retry_request["service_tier"] = FALLBACK_OPENAI_SERVICE_TIER
        return (
            "service_tier_auto",
            retry_request,
            "OpenAI Flex processing unavailable for this request; retrying with standard processing (service_tier=auto).",
        )
    return None, None, None


def _should_retry_with_max_completion_tokens(exc: Exception, request: dict[str, Any]) -> bool:
    if "max_tokens" not in request or "max_completion_tokens" in request:
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code not in (400, 422):
        return False
    lowered = _extract_error_text(exc).lower()
    return "max_tokens" in lowered and "max_completion_tokens" in lowered and any(
        marker in lowered for marker in ("unsupported", "not supported", "unsupported_parameter")
    )


def _should_retry_with_standard_processing(exc: Exception, requested_tier: str | None) -> bool:
    if requested_tier != DEFAULT_OPENAI_SERVICE_TIER:
        return False
    status_code = getattr(exc, "status_code", None)
    error_text = _extract_error_text(exc)
    lowered = error_text.lower()
    if status_code == 429 and (
        "resource unavailable" in lowered or "resource_unavailable" in lowered
    ):
        return True
    if status_code in (400, 422) and "service_tier" in lowered and "flex" in lowered:
        if any(
            marker in lowered
            for marker in (
                "unsupported",
                "unsupported_value",
                "not available",
                "not supported",
                "invalid",
                "beta",
            )
        ):
            return True
    return False


def _extract_error_text(exc: Exception) -> str:
    parts: list[str] = [str(exc)]
    for attr in ("message", "code", "type"):
        value = getattr(exc, attr, None)
        if value:
            parts.append(str(value))
    body = getattr(exc, "body", None)
    _append_nested_error_text(parts, body)
    response = getattr(exc, "response", None)
    if response is not None:
        _append_nested_error_text(parts, getattr(response, "json", None))
        _append_nested_error_text(parts, getattr(response, "text", None))
    return " ".join(part for part in parts if part)


def _append_nested_error_text(parts: list[str], value: Any) -> None:
    if not value:
        return
    if isinstance(value, dict):
        for nested in value.values():
            _append_nested_error_text(parts, nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _append_nested_error_text(parts, nested)
        return
    if callable(value):
        return
    parts.append(str(value))
