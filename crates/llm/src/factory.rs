//! Shared construction of OpenAI-compatible LLM adapters.
//!
//! The embedded component manager (`cognee-lib`) and the standalone HTTP server
//! (`cognee-http-server`) both wire the LLM the same way: an [`OpenAIAdapter`]
//! built from the configured model / key / endpoint, with structured-output and
//! network retries applied. Centralising that here keeps the two wiring paths in
//! sync (see issue #17). Provider routing beyond `openai` is layered on top of
//! this function in a follow-up; today it covers the OpenAI / OpenAI-compatible
//! path that both call sites already used.

use crate::{LlmError, LlmResult, OpenAIAdapter};

/// Build an [`OpenAIAdapter`] for an OpenAI-compatible `provider`.
///
/// `endpoint` is the raw configured value; an empty or whitespace-only string is
/// treated as "unset" so the adapter falls back to its provider default.
/// `max_retries` is floored at 1 and applied to both the structured-output and
/// network retry loops, matching the previous inline wiring.
///
/// Returns [`LlmError::ConfigError`] when a required API key is missing or the
/// provider is unsupported, so each caller can decide whether to hard-fail
/// (component manager) or skip and wire `None` (HTTP server).
pub fn build_openai_compatible_adapter(
    provider: &str,
    model: &str,
    api_key: &str,
    endpoint: &str,
    max_retries: u32,
) -> LlmResult<OpenAIAdapter> {
    let retries = max_retries.max(1);

    match provider.to_ascii_lowercase().as_str() {
        "openai" => {
            if api_key.is_empty() {
                return Err(LlmError::ConfigError(
                    "llm_api_key must be configured".to_string(),
                ));
            }

            let adapter = OpenAIAdapter::new(
                model.to_string(),
                api_key.to_string(),
                endpoint_or_default(endpoint),
            )?
            .with_structured_output_retries(retries)
            .with_network_retries(retries);
            Ok(adapter)
        }
        other => Err(LlmError::ConfigError(format!(
            "Unsupported llm_provider '{other}'. Supported: openai."
        ))),
    }
}

/// Treat an empty / whitespace endpoint as "unset" so the adapter uses its
/// provider default base URL.
fn endpoint_or_default(endpoint: &str) -> Option<String> {
    if endpoint.trim().is_empty() {
        None
    } else {
        Some(endpoint.to_string())
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
}
