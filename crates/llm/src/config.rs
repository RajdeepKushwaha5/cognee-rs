//! LLM configuration.

use serde::{Deserialize, Serialize};

/// LLM provider type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmProvider {
    OpenAI,
    LiteRt,
    Anthropic,
    Ollama,
    Gemini,
    Mistral,
    Bedrock,
    Local,
    /// Record/replay cassette-based mock LLM (selected via `MOCK_LLM`).
    Mock,
}

/// LLM configuration.
///
/// `Debug` is hand-written so `api_key` / `fallback_api_key` never reach a log
/// line: each prints as `***REDACTED***` when set and `None` when unset.
#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// LLM provider.
    pub provider: LlmProvider,

    /// Model identifier (e.g., "gpt-4", "claude-3-opus").
    pub model: String,

    /// API key (if required by provider).
    pub api_key: Option<String>,

    /// API endpoint (custom endpoint for self-hosted models).
    pub endpoint: Option<String>,

    /// Default temperature.
    pub temperature: f32,

    /// Default max tokens.
    pub max_tokens: u32,

    /// Enable streaming responses.
    pub streaming: bool,

    /// Request timeout in seconds.
    pub timeout_seconds: u64,

    /// Maximum number of retries.
    pub max_retries: u32,

    /// Enable rate limiting.
    pub rate_limit_enabled: bool,

    /// Rate limit: requests per interval.
    pub rate_limit_requests: u32,

    /// Rate limit interval in seconds.
    pub rate_limit_interval_seconds: u64,

    /// Fallback model (if primary fails).
    pub fallback_model: Option<String>,

    /// Fallback API key.
    pub fallback_api_key: Option<String>,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use cognee_config_core::{CredentialSafe, Redacted};
        // Exhaustive: a new field fails to compile until it is shown here.
        let Self {
            provider,
            model,
            api_key,
            endpoint,
            temperature,
            max_tokens,
            streaming,
            timeout_seconds,
            max_retries,
            rate_limit_enabled,
            rate_limit_requests,
            rate_limit_interval_seconds,
            fallback_model,
            fallback_api_key,
        } = self;
        f.debug_struct("LlmConfig")
            .field("provider", provider)
            .field("model", model)
            .field("api_key", &Redacted(api_key))
            .field("endpoint", &CredentialSafe(endpoint))
            .field("temperature", temperature)
            .field("max_tokens", max_tokens)
            .field("streaming", streaming)
            .field("timeout_seconds", timeout_seconds)
            .field("max_retries", max_retries)
            .field("rate_limit_enabled", rate_limit_enabled)
            .field("rate_limit_requests", rate_limit_requests)
            .field("rate_limit_interval_seconds", rate_limit_interval_seconds)
            .field("fallback_model", fallback_model)
            .field("fallback_api_key", &Redacted(fallback_api_key))
            .finish()
    }
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::OpenAI,
            model: "gpt-4".to_string(),
            api_key: None,
            endpoint: None,
            temperature: 0.0,
            max_tokens: 16384,
            streaming: false,
            timeout_seconds: 120,
            max_retries: 3,
            rate_limit_enabled: false,
            rate_limit_requests: 60,
            rate_limit_interval_seconds: 60,
            fallback_model: None,
            fallback_api_key: None,
        }
    }
}

impl LlmConfig {
    /// Create a new configuration with minimal required fields.
    pub fn new(provider: LlmProvider, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set custom endpoint.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set timeout.
    pub fn with_timeout_seconds(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Set max retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `{:?}` never prints either API key.
    #[test]
    fn debug_redacts_api_keys() {
        let config = LlmConfig {
            api_key: Some("sk-primary-secret".to_string()),
            fallback_api_key: Some("sk-fallback-secret".to_string()),
            ..LlmConfig::default()
        };
        let out = format!("{config:?}");
        assert!(!out.contains("sk-primary-secret"), "{out}");
        assert!(!out.contains("sk-fallback-secret"), "{out}");
        assert!(out.contains("api_key: ***REDACTED***"), "{out}");
        assert!(out.contains("model: \"gpt-4\""), "{out}");
        assert!(format!("{:?}", LlmConfig::default()).contains("fallback_api_key: None"));
    }
}
