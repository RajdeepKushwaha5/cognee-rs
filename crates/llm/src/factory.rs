//! Shared construction of OpenAI-compatible LLM adapters.
//!
//! The embedded component manager (`cognee-lib`) and the standalone HTTP server
//! (`cognee-http-server`) both wire the LLM the same way: an [`OpenAIAdapter`]
//! built from the configured model / key / endpoint, with structured-output and
//! network retries applied. Centralising that here keeps the two wiring paths in
//! sync (see issue #17).
//!
//! Several providers in the Python SDK are themselves OpenAI-compatible HTTP
//! endpoints, so they need only factory routing — not a new adapter. This module
//! routes `openai`, `ollama`, `mistral`, `gemini`, and `custom` /
//! `openai_compatible` onto the same [`OpenAIAdapter`], differing only in the base
//! URL, whether an API key is mandatory, and litellm-style model-prefix stripping.
//! The OpenAI-only request quirks in the adapter are gated on the
//! `api.openai.com` host, so pointing it at another compatible endpoint does not
//! trigger any OpenAI-specific behaviour.

use crate::{LlmError, LlmResult, OpenAIAdapter};

/// Default OpenAI-compatible base URL for a local Ollama server.
const OLLAMA_DEFAULT_ENDPOINT: &str = "http://localhost:11434/v1";
/// Default Mistral OpenAI-compatible base URL.
const MISTRAL_DEFAULT_ENDPOINT: &str = "https://api.mistral.ai/v1";
/// Default Gemini OpenAI-compatible base URL.
const GEMINI_DEFAULT_ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta/openai/";

/// Build an [`OpenAIAdapter`] for an OpenAI-compatible `provider`.
///
/// Supported providers (case-insensitive): `openai`, `ollama`, `mistral`,
/// `gemini`, and `custom` / `openai_compatible`.
///
/// `endpoint` is the raw configured value; an empty or whitespace-only string is
/// treated as "unset" so the provider's default base URL is used (or, for
/// `openai`, the adapter's built-in OpenAI default). `custom` /
/// `openai_compatible` has no default and requires an explicit endpoint.
///
/// `max_retries` is floored at 1 and applied to both the structured-output and
/// network retry loops, matching the previous inline wiring.
///
/// litellm-style provider prefixes on the model (`ollama/`, `mistral/`,
/// `gemini/`) are stripped so provider-qualified config values keep working
/// (the adapter itself only strips `openai/`).
///
/// Returns [`LlmError::ConfigError`] when a required API key or endpoint is
/// missing, or the provider is unsupported, so each caller can decide whether to
/// hard-fail (component manager) or skip and wire `None` (HTTP server).
pub fn build_openai_compatible_adapter(
    provider: &str,
    model: &str,
    api_key: &str,
    endpoint: &str,
    max_retries: u32,
) -> LlmResult<OpenAIAdapter> {
    let retries = max_retries.max(1);
    let provider = provider.to_ascii_lowercase();
    let endpoint = endpoint.trim();

    // Per provider: resolved base URL, whether an API key is mandatory, and the
    // litellm-style model prefix to strip (if any).
    let (base_url, api_key_required, strip_prefix): (Option<String>, bool, Option<&str>) =
        match provider.as_str() {
            // Empty endpoint → None so the adapter uses its OpenAI default.
            "openai" => (non_empty(endpoint), true, None),
            "ollama" => (
                Some(endpoint_or(endpoint, OLLAMA_DEFAULT_ENDPOINT)),
                false,
                Some("ollama/"),
            ),
            "mistral" => (
                Some(endpoint_or(endpoint, MISTRAL_DEFAULT_ENDPOINT)),
                true,
                Some("mistral/"),
            ),
            "gemini" => (
                Some(endpoint_or(endpoint, GEMINI_DEFAULT_ENDPOINT)),
                true,
                Some("gemini/"),
            ),
            "custom" | "openai_compatible" => {
                if endpoint.is_empty() {
                    return Err(LlmError::ConfigError(format!(
                        "llm_endpoint must be configured for provider '{provider}'"
                    )));
                }
                (Some(endpoint.to_string()), false, None)
            }
            other => {
                return Err(LlmError::ConfigError(format!(
                    "Unsupported llm_provider '{other}'. \
                     Supported: openai, ollama, mistral, gemini, custom."
                )));
            }
        };

    if api_key_required && api_key.is_empty() {
        return Err(LlmError::ConfigError(
            "llm_api_key must be configured".to_string(),
        ));
    }

    // Ollama ignores auth but the OpenAI-style client still sends a bearer token;
    // use a harmless placeholder when none is configured (matches Python cognee's
    // `LLM_API_KEY="ollama"` convention).
    let api_key = if provider == "ollama" && api_key.is_empty() {
        "ollama"
    } else {
        api_key
    };

    let model = match strip_prefix {
        Some(prefix) => model.strip_prefix(prefix).unwrap_or(model),
        None => model,
    };

    let adapter = OpenAIAdapter::new(model.to_string(), api_key.to_string(), base_url)?
        .with_structured_output_retries(retries)
        .with_network_retries(retries);
    Ok(adapter)
}

/// `Some(endpoint)` unless it is empty (already trimmed by the caller).
fn non_empty(endpoint: &str) -> Option<String> {
    if endpoint.is_empty() {
        None
    } else {
        Some(endpoint.to_string())
    }
}

/// The configured endpoint, or `default` when it is empty (already trimmed).
fn endpoint_or(endpoint: &str, default: &str) -> String {
    if endpoint.is_empty() {
        default.to_string()
    } else {
        endpoint.to_string()
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;
    use crate::Llm;

    #[test]
    fn builds_openai_adapter_and_strips_provider_prefix() {
        let adapter =
            build_openai_compatible_adapter("openai", "openai/gpt-4o-mini", "sk-test", "", 3)
                .expect("adapter should build");
        // OpenAIAdapter strips the leading `openai/` litellm-style prefix.
        assert_eq!(adapter.model(), "gpt-4o-mini");
    }

    #[test]
    fn openai_requires_api_key() {
        let result = build_openai_compatible_adapter("openai", "gpt-4o-mini", "", "", 3);
        assert!(matches!(result, Err(LlmError::ConfigError(_))));
    }

    #[test]
    fn provider_matching_is_case_insensitive() {
        let adapter = build_openai_compatible_adapter("OpenAI", "gpt-4o-mini", "sk-test", "", 1)
            .expect("adapter should build");
        assert_eq!(adapter.model(), "gpt-4o-mini");
    }

    #[test]
    fn unsupported_provider_errors() {
        let result = build_openai_compatible_adapter("acme", "model", "key", "", 3);
        assert!(matches!(result, Err(LlmError::ConfigError(_))));
    }

    #[test]
    fn ollama_defaults_endpoint_and_allows_empty_key() {
        // No endpoint, no key → still builds (Ollama needs neither from the user).
        let adapter = build_openai_compatible_adapter("ollama", "ollama/llama3.1:8b", "", "", 3)
            .expect("ollama adapter should build");
        assert_eq!(adapter.model(), "llama3.1:8b");
    }

    #[test]
    fn ollama_honors_custom_endpoint() {
        let adapter = build_openai_compatible_adapter(
            "ollama",
            "llama3.1:8b",
            "",
            "http://remote:11434/v1",
            3,
        )
        .expect("ollama adapter should build");
        assert_eq!(adapter.model(), "llama3.1:8b");
    }

    #[test]
    fn mistral_requires_key_and_strips_prefix() {
        let missing =
            build_openai_compatible_adapter("mistral", "mistral/mistral-large", "", "", 3);
        assert!(matches!(missing, Err(LlmError::ConfigError(_))));

        let adapter = build_openai_compatible_adapter(
            "mistral",
            "mistral/mistral-large-latest",
            "sk-test",
            "",
            3,
        )
        .expect("mistral adapter should build");
        assert_eq!(adapter.model(), "mistral-large-latest");
    }

    #[test]
    fn gemini_requires_key_and_strips_prefix() {
        let adapter =
            build_openai_compatible_adapter("gemini", "gemini/gemini-2.0-flash", "sk-test", "", 3)
                .expect("gemini adapter should build");
        assert_eq!(adapter.model(), "gemini-2.0-flash");
    }

    #[test]
    fn custom_requires_endpoint() {
        let missing = build_openai_compatible_adapter("custom", "my-model", "", "", 3);
        assert!(matches!(missing, Err(LlmError::ConfigError(_))));

        let adapter = build_openai_compatible_adapter(
            "openai_compatible",
            "my-model",
            "",
            "https://my.host/v1",
            3,
        )
        .expect("custom adapter should build");
        // No prefix stripping for custom — the model is passed through verbatim.
        assert_eq!(adapter.model(), "my-model");
    }
}
