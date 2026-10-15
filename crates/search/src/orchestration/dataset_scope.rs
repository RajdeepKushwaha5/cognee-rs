use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::types::{SearchContext, SearchItem};

pub fn scope_context_by_datasets(
    context: &SearchContext,
    dataset_ids: &[Uuid],
) -> HashMap<String, SearchContext> {
    let mut scoped_contexts: HashMap<String, SearchContext> = dataset_ids
        .iter()
        .map(|dataset_id| (dataset_id.to_string(), Vec::new()))
        .collect();

    for item in context {
        for dataset_id in item_dataset_ids(item) {
            if let Some(scoped_items) = scoped_contexts.get_mut(dataset_id) {
                scoped_items.push(item.clone());
            }
        }
    }

    scoped_contexts
}

/// The datasets a context item belongs to.
///
/// A content-addressed point can belong to several datasets; its full
/// membership lives in the `dataset_ids` array (unioned on upsert). When
/// present it is authoritative. Fall back to the scalar `dataset_id` only for
/// points written before the array existed (back-compat). An item carrying
/// neither belongs to no dataset, so no dataset scope admits it.
fn item_dataset_ids(item: &SearchItem) -> Vec<&str> {
    match item
        .payload
        .get("dataset_ids")
        .and_then(|value| value.as_array())
    {
        Some(ids) => ids.iter().filter_map(|v| v.as_str()).collect(),
        None => item
            .payload
            .get("dataset_id")
            .and_then(|value| value.as_str())
            .into_iter()
            .collect(),
    }
}

/// `context` narrowed to the items that belong to at least one of
/// `dataset_ids`, in the retriever's original (ranked) order, each item once.
///
/// The order-preserving counterpart of `merge_scoped_contexts(
/// &scope_context_by_datasets(context, dataset_ids))`, whose `HashMap`
/// iteration loses the ranking. Membership is decided exactly as
/// [`scope_context_by_datasets`] decides it.
pub fn filter_context_to_datasets(context: &SearchContext, dataset_ids: &[Uuid]) -> SearchContext {
    let allowed: HashSet<String> = dataset_ids.iter().map(Uuid::to_string).collect();
    context
        .iter()
        .filter(|item| {
            item_dataset_ids(item)
                .into_iter()
                .any(|id| allowed.contains(id))
        })
        .cloned()
        .collect()
}

/// `context` narrowed to `params.dataset_ids` when the request has a dataset
/// scope; unchanged otherwise.
///
/// For retrievers that fetch **more** context inside `get_completion`
/// (context extension, chain-of-thought follow-ups): the orchestrator filters
/// only the context it fetches itself, so a follow-up retrieval must apply the
/// scope on its own or the answer draws on datasets outside it.
pub(crate) fn filter_context_to_params(
    context: SearchContext,
    params: &crate::types::SearchParams,
) -> SearchContext {
    match params.dataset_ids.as_deref() {
        Some(scope) if !scope.is_empty() => filter_context_to_datasets(&context, scope),
        _ => context,
    }
}

pub fn merge_scoped_contexts(scoped_contexts: &HashMap<String, SearchContext>) -> SearchContext {
    let mut seen = HashSet::new();
    let mut merged = Vec::new();

    for context in scoped_contexts.values() {
        for item in context {
            let dedup_key = dedup_key(item);
            if seen.insert(dedup_key) {
                merged.push(item.clone());
            }
        }
    }

    merged
}

fn dedup_key(item: &SearchItem) -> String {
    match item.id {
        Some(id) => id.to_string(),
        None => item.payload.to_string(),
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use serde_json::json;
    use uuid::Uuid;

    use crate::types::SearchItem;

    #[test]
    fn scopes_context_by_dataset_id_payload() {
        let dataset_a = Uuid::new_v4();
        let dataset_b = Uuid::new_v4();

        let context = vec![
            SearchItem {
                id: None,
                score: Some(0.8),
                payload: json!({"dataset_id": dataset_a.to_string(), "text": "A1"}),
            },
            SearchItem {
                id: None,
                score: Some(0.7),
                payload: json!({"dataset_id": dataset_b.to_string(), "text": "B1"}),
            },
        ];

        let scoped = super::scope_context_by_datasets(&context, &[dataset_a, dataset_b]);

        assert_eq!(scoped.get(&dataset_a.to_string()).unwrap().len(), 1);
        assert_eq!(scoped.get(&dataset_b.to_string()).unwrap().len(), 1);
        assert_eq!(scoped[&dataset_a.to_string()][0].payload["text"], "A1");
        assert_eq!(scoped[&dataset_b.to_string()][0].payload["text"], "B1");
    }

    #[test]
    fn scopes_shared_point_to_all_member_datasets() {
        // A content-addressed point that belongs to two datasets (its
        // `dataset_ids` union array lists both) must be assigned to BOTH scopes,
        // not just one. Regression test for the cross-dataset dedup bug.
        let dataset_a = Uuid::new_v4();
        let dataset_b = Uuid::new_v4();

        let context = vec![SearchItem {
            id: Some(Uuid::new_v4()),
            score: Some(0.9),
            payload: json!({
                "dataset_ids": [dataset_a.to_string(), dataset_b.to_string()],
                "dataset_id": dataset_b.to_string(),
                "text": "shared across A and B",
            }),
        }];

        let scoped = super::scope_context_by_datasets(&context, &[dataset_a, dataset_b]);

        assert_eq!(
            scoped.get(&dataset_a.to_string()).unwrap().len(),
            1,
            "shared point must be retrievable when scoped to dataset A"
        );
        assert_eq!(
            scoped.get(&dataset_b.to_string()).unwrap().len(),
            1,
            "shared point must be retrievable when scoped to dataset B"
        );
    }

    #[test]
    fn merges_scoped_contexts_without_duplicates() {
        let dataset_a = Uuid::new_v4().to_string();
        let dataset_b = Uuid::new_v4().to_string();
        let duplicate_id = Uuid::new_v4();

        let item_shared = SearchItem {
            id: Some(duplicate_id),
            score: Some(0.9),
            payload: json!({"dataset_id": dataset_a, "text": "shared"}),
        };

        let item_unique = SearchItem {
            id: None,
            score: Some(0.7),
            payload: json!({"dataset_id": dataset_b, "text": "unique"}),
        };

        let merged = super::merge_scoped_contexts(&std::collections::HashMap::from([
            ("a".to_string(), vec![item_shared.clone()]),
            ("b".to_string(), vec![item_shared, item_unique]),
        ]));

        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn filter_context_to_datasets_keeps_ranked_order_and_membership() {
        let dataset_a = Uuid::new_v4();
        let dataset_b = Uuid::new_v4();
        let dataset_c = Uuid::new_v4();
        let item = |payload| SearchItem {
            id: None,
            score: None,
            payload,
        };
        let context = vec![
            item(json!({"dataset_id": dataset_b.to_string(), "text": "B"})),
            item(json!({"dataset_id": dataset_c.to_string(), "text": "C"})),
            item(
                json!({"dataset_ids": [dataset_c.to_string(), dataset_a.to_string()], "text": "CA"}),
            ),
            item(json!({"text": "none"})),
            item(json!({"dataset_id": dataset_a.to_string(), "text": "A"})),
        ];

        let filtered = super::filter_context_to_datasets(&context, &[dataset_a, dataset_b]);
        let texts: Vec<&str> = filtered
            .iter()
            .map(|i| i.payload["text"].as_str().unwrap())
            .collect();
        assert_eq!(texts, vec!["B", "CA", "A"]);
    }

    #[test]
    fn filter_context_to_params_applies_only_a_non_empty_scope() {
        let dataset_a = Uuid::new_v4();
        let context = vec![
            SearchItem {
                id: None,
                score: None,
                payload: json!({"dataset_id": dataset_a.to_string(), "text": "A"}),
            },
            SearchItem {
                id: None,
                score: None,
                payload: json!({"dataset_id": Uuid::new_v4().to_string(), "text": "other"}),
            },
        ];
        let scoped = crate::types::SearchParams {
            dataset_ids: Some(vec![dataset_a]),
            ..Default::default()
        };
        assert_eq!(
            super::filter_context_to_params(context.clone(), &scoped).len(),
            1
        );
        for unscoped in [
            crate::types::SearchParams::default(),
            crate::types::SearchParams {
                dataset_ids: Some(vec![]),
                ..Default::default()
            },
        ] {
            assert_eq!(
                super::filter_context_to_params(context.clone(), &unscoped).len(),
                2
            );
        }
    }
}
