use cognee_vector::{SearchResult, VectorDB};

use crate::types::{SearchContext, SearchError, SearchItem};

pub(crate) fn search_results_to_context(
    results: Vec<SearchResult>,
) -> Result<SearchContext, SearchError> {
    results
        .into_iter()
        .map(|result| {
            let payload = serde_json::to_value(result.metadata)?;
            Ok(SearchItem {
                id: Some(result.id),
                score: Some(result.score),
                payload,
            })
        })
        .collect()
}

/// Similarity-search one collection, scoped to a node set when `node_name` is
/// non-empty.
///
/// Python threads `node_name` / `node_name_filter_operator` into
/// `vector_engine.search(...)` for the chunk, RAG and hybrid chunk lanes
/// (`chunks_retriever.py:127-135`, `completion_retriever.py:136-144`,
/// `hybrid/chunks.py:143-175`): the engine filters server-side, then limits,
/// so in-set rows are never crowded out by out-of-set ones. With no node set
/// this is the plain `search_similar` call.
pub(crate) async fn search_node_filtered(
    vector_db: &dyn VectorDB,
    data_type: &str,
    field: &str,
    query_vector: &[f32],
    limit: usize,
    node_name: Option<&[String]>,
    node_name_filter_operator: &str,
) -> Result<SearchContext, SearchError> {
    let results = match node_name {
        Some(names) if !names.is_empty() => {
            vector_db
                .search_similar_filtered(
                    data_type,
                    field,
                    query_vector,
                    limit,
                    Some(names),
                    node_name_filter_operator,
                )
                .await?
        }
        _ => {
            vector_db
                .search_similar(data_type, field, query_vector, limit)
                .await?
        }
    };
    search_results_to_context(results)
}
