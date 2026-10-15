use std::sync::Arc;

use async_trait::async_trait;
use cognee_llm::Message;
use cognee_session::SessionContext;

use crate::types::{SearchContext, SearchError, SearchOutput, SearchParams, SearchType};

pub type SearchRetrieverRef = Arc<dyn SearchRetriever>;

/// The [`SearchTuning`](crate::orchestration::SearchTuning) knobs a retriever
/// was built with, for tests that check `SearchBuilder` wires the tuning into
/// every retriever that consumes it. `None` means "this retriever has no such
/// knob".
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AppliedTuning {
    /// `search.triplet_neighbor_cap` (itself optional: `None` is uncapped).
    pub neighbor_cap: Option<Option<usize>>,
    /// `search.allow_cypher_query`.
    pub allow_cypher_query: Option<bool>,
    /// The `search.hybrid` budget.
    pub hybrid: Option<super::HybridBudget>,
}

#[async_trait]
pub trait SearchRetriever: Send + Sync {
    fn search_type(&self) -> SearchType;

    /// The tuning knobs this retriever was built with (test-only
    /// introspection; see [`AppliedTuning`]).
    #[cfg(test)]
    fn applied_tuning(&self) -> AppliedTuning {
        AppliedTuning::default()
    }

    async fn get_context(
        &self,
        query: &str,
        params: &SearchParams,
    ) -> Result<SearchContext, SearchError>;

    async fn get_completion(
        &self,
        query: &str,
        context: Option<SearchContext>,
        session: &SessionContext,
        params: &SearchParams,
    ) -> Result<SearchOutput, SearchError>;

    /// The messages [`get_completion`](Self::get_completion) would send the
    /// LLM for `context`, built without calling it — what an `only_context`
    /// search returns in place of the bare context.
    ///
    /// `None` when this retriever sends no single templated prompt (the
    /// non-generative types: chunks, summaries, lexical, Cypher, natural
    /// language, coding rules, feedback) or when the rendered context is
    /// empty, which is also when `get_completion` skips the LLM. Python's
    /// `build_only_context_prompt` (`only_context_prompt.py`) returns `None`
    /// in the same two cases.
    ///
    /// Retrievers whose context stage calls the LLM — graph summary's
    /// summarization, chain-of-thought and context-extension rounds, hybrid's
    /// overflow summaries — make those calls here too, as Python's preview
    /// does; only the answer itself is never generated.
    ///
    /// Implementations must build the messages with the same code their
    /// `get_completion` uses, so the preview cannot drift from the real call.
    async fn completion_messages(
        &self,
        _query: &str,
        _context: &SearchContext,
        _session: &SessionContext,
        _params: &SearchParams,
    ) -> Result<Option<Vec<Message>>, SearchError> {
        Ok(None)
    }

    /// Process multiple queries in sequence and return their contexts.
    ///
    /// Default: loops over [`get_context`]. Override for efficient batching.
    async fn get_context_batch(
        &self,
        queries: &[String],
        params: &SearchParams,
    ) -> Result<Vec<SearchContext>, SearchError> {
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.get_context(query, params).await?);
        }
        Ok(results)
    }

    /// Process multiple queries and return their completions.
    ///
    /// Default: loops over [`get_completion`]. Override for efficient batching.
    async fn get_completion_batch(
        &self,
        queries: &[String],
        contexts: Option<Vec<SearchContext>>,
        session: &SessionContext,
        params: &SearchParams,
    ) -> Result<Vec<SearchOutput>, SearchError> {
        let mut results = Vec::with_capacity(queries.len());
        for (i, query) in queries.iter().enumerate() {
            let ctx = contexts.as_ref().and_then(|cs| cs.get(i).cloned());
            results.push(self.get_completion(query, ctx, session, params).await?);
        }
        Ok(results)
    }
}
