from __future__ import annotations

import re
from dataclasses import dataclass


DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_OLLAMA = "qwen3:14b"
DEFAULT_PIPELINE_ARCHITECTURE = "three_layer"
DEFAULT_PIPELINE_ARCHITECTURE_SELECTION = "auto"
PIPELINE_ARCHITECTURE_CHOICES = ("auto", "three_layer", "one_pass")
DEFAULT_OLLAMA_NUM_CTX = 8192
LARGE_CTX_THRESHOLD = 32768
DEFAULT_SINGLE_PASS_CHUNK_CHARS = 600
DEFAULT_MAP_CHUNK_CHARS = 4000


@dataclass(frozen=True)
class ModelProfile:
    model: str
    provider: str
    num_ctx: int | None = None
    estimated_vram_gb: int | None = None
    preferred_pipeline_architecture: str = DEFAULT_PIPELINE_ARCHITECTURE
    chunk_chars: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model: str
    provider: str
    architecture: str
    num_ctx: int | None = None
    chunk_chars: int = DEFAULT_MAP_CHUNK_CHARS
    estimated_vram_gb: int | None = None


KNOWN_MODEL_PROFILES: dict[tuple[str, str], ModelProfile] = {
    ("ollama", "qwen2.5:7b-32k"): ModelProfile(
        model="qwen2.5:7b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=6,
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Legacy baseline local model used for the initial 3-layer evaluation.",
    ),
    ("ollama", "qwen2.5:14b-32k"): ModelProfile(
        model="qwen2.5:14b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Step 3.1 default upgrade candidate; best local quality/resource balance.",
    ),
    ("ollama", "qwen3:14b"): ModelProfile(
        model="qwen3:14b",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Current default local model for Step 3.1.",
    ),
    ("ollama", "qwen3:14b-32k"): ModelProfile(
        model="qwen3:14b-32k",
        provider="ollama",
        num_ctx=32768,
        estimated_vram_gb=10,
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Newer 14B architecture to benchmark against qwen2.5:14b.",
    ),
    ("ollama", "gemma3:27b"): ModelProfile(
        model="gemma3:27b",
        provider="ollama",
        num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        estimated_vram_gb=18,
        preferred_pipeline_architecture="one_pass",
        chunk_chars=DEFAULT_MAP_CHUNK_CHARS,
        notes="Large multilingual local model candidate for Step 3.1.",
    ),
    ("ollama", "llama3.3:70b"): ModelProfile(
        model="llama3.3:70b",
        provider="ollama",
        num_ctx=DEFAULT_OLLAMA_NUM_CTX,
        estimated_vram_gb=40,
        preferred_pipeline_architecture="one_pass",
        chunk_chars=DEFAULT_MAP_CHUNK_CHARS,
        notes="Very large fallback candidate if hardware allows.",
    ),
    ("openai", DEFAULT_MODEL_OPENAI): ModelProfile(
        model=DEFAULT_MODEL_OPENAI,
        provider="openai",
        preferred_pipeline_architecture="one_pass",
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Reference OpenAI baseline used only when explicitly selected.",
    ),
    ("openai", "gpt-5.4-mini"): ModelProfile(
        model="gpt-5.4-mini",
        provider="openai",
        preferred_pipeline_architecture="one_pass",
        chunk_chars=DEFAULT_SINGLE_PASS_CHUNK_CHARS,
        notes="Lower-cost GPT-5.4 variant for benchmark comparisons.",
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
_MODEL_SIZE_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)b(?![a-z0-9])", re.IGNORECASE)


def normalize_model_name(model: str) -> str:
    model = (model or "").strip()
    if model.endswith(":latest") and model.count(":") >= 1:
        model = model[: -len(":latest")]
    return model


def get_model_profile(provider: str, model: str) -> ModelProfile | None:
    key = (provider.strip().lower(), normalize_model_name(model))
    return KNOWN_MODEL_PROFILES.get(key)


def infer_model_size_billions(model: str) -> float | None:
    normalized = normalize_model_name(model)
    match = _MODEL_SIZE_RE.search(normalized)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


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


def infer_preferred_pipeline_architecture(provider: str, model: str) -> str:
    provider_norm = provider.strip().lower()
    profile = get_model_profile(provider_norm, model)
    if profile:
        return profile.preferred_pipeline_architecture
    if provider_norm != "ollama":
        return "one_pass"
    model_size = infer_model_size_billions(model)
    if model_size is not None and model_size >= 27:
        return "one_pass"
    return DEFAULT_PIPELINE_ARCHITECTURE


def resolve_pipeline_architecture(provider: str, model: str, requested_architecture: str | None = None) -> str:
    requested = (requested_architecture or "").strip().lower()
    if requested and requested != DEFAULT_PIPELINE_ARCHITECTURE_SELECTION:
        return requested
    return infer_preferred_pipeline_architecture(provider, model)


def infer_topic_chunk_chars(provider: str, model: str) -> int:
    provider_norm = provider.strip().lower()
    profile = get_model_profile(provider_norm, model)
    if profile and profile.chunk_chars:
        return profile.chunk_chars
    if model_supports_large_session_single_pass(provider_norm, model):
        return DEFAULT_SINGLE_PASS_CHUNK_CHARS
    return DEFAULT_MAP_CHUNK_CHARS


def get_model_runtime_config(provider: str, model: str) -> ModelRuntimeConfig:
    provider_norm = provider.strip().lower()
    normalized_model = normalize_model_name(model)
    profile = get_model_profile(provider_norm, normalized_model)
    if provider_norm == "ollama":
        num_ctx = infer_ollama_num_ctx(normalized_model)
    else:
        num_ctx = profile.num_ctx if profile else None
    return ModelRuntimeConfig(
        model=normalized_model,
        provider=provider_norm,
        architecture=infer_preferred_pipeline_architecture(provider_norm, normalized_model),
        num_ctx=num_ctx,
        chunk_chars=infer_topic_chunk_chars(provider_norm, normalized_model),
        estimated_vram_gb=profile.estimated_vram_gb if profile else None,
    )
