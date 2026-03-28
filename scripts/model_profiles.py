from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "qwen3:14b"
DEFAULT_PIPELINE_ARCHITECTURE = "three_layer"
DEFAULT_OLLAMA_NUM_CTX = 8192
LARGE_CTX_THRESHOLD = 32768


@dataclass(frozen=True)
class ModelProfile:
    model: str
    provider: str
    num_ctx: int | None = None
    estimated_vram_gb: int | None = None
    preferred_pipeline_architecture: str = DEFAULT_PIPELINE_ARCHITECTURE
    notes: str = ""


KNOWN_MODEL_PROFILES: dict[tuple[str, str], ModelProfile] = {
    ("ollama", "qwen2.5:7b-32k"): ModelProfile(
        model="qwen2.5:7b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=6,
        notes="Legacy baseline local model used for the initial 3-layer evaluation.",
    ),
    ("ollama", "qwen2.5:14b-32k"): ModelProfile(
        model="qwen2.5:14b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        notes="Step 3.1 default upgrade candidate; best local quality/resource balance.",
    ),
    ("ollama", "qwen3:14b"): ModelProfile(
        model="qwen3:14b",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        notes="Current default local model for Step 3.1.",
    ),
    ("ollama", "qwen3:14b-32k"): ModelProfile(
        model="qwen3:14b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        notes="Newer 14B architecture to benchmark against qwen2.5:14b.",
    ),
    ("ollama", "gemma3:27b"): ModelProfile(
        model="gemma3:27b",
        provider="ollama",
        num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        estimated_vram_gb=18,
        notes="Large multilingual local model candidate for Step 3.1.",
    ),
    ("ollama", "llama3.3:70b"): ModelProfile(
        model="llama3.3:70b",
        provider="ollama",
        num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        estimated_vram_gb=40,
        notes="Very large fallback candidate if hardware allows.",
    ),
    ("openai", DEFAULT_MODEL_OPENAI): ModelProfile(
        model=DEFAULT_MODEL_OPENAI,
        provider="openai",
        preferred_pipeline_architecture="one_pass",
        notes="Reference OpenAI baseline used only when explicitly selected.",
    ),
}

STEP_3_1_CANDIDATE_MODELS = (
    "qwen3:14b",
    "qwen2.5:14b-32k",
    "qwen3:14b-32k",
    "gemma3:27b",
    "llama3.3:70b",
)

_CTX_SUFFIX_RE = re.compile(r"(?<!\d)(\d{1,3})k(?![a-z0-9])", re.IGNORECASE)


def normalize_model_name(model: str) -> str:
    model = (model or "").strip()
    if model.endswith(":latest") and model.count(":") >= 1:
        model = model[: -len(":latest")]
    return model


def get_model_profile(provider: str, model: str) -> ModelProfile | None:
    key = (provider.strip().lower(), normalize_model_name(model))
    return KNOWN_MODEL_PROFILES.get(key)


def infer_ollama_num_ctx(model: str) -> int:
    profile = get_model_profile("ollama", model)
    if profile and profile.num_ctx:
        return profile.num_ctx

    normalized = normalize_model_name(model)
    match = _CTX_SUFFIX_RE.search(normalized)
    if match:
        return int(match.group(1)) * 1024
    return DEFAULT_OLLAMA_NUM_CTX


def model_supports_large_session_single_pass(provider: str, model: str) -> bool:
    if provider.strip().lower() != "ollama":
        return True
    return infer_ollama_num_ctx(model) >= LARGE_CTX_THRESHOLD
