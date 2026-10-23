#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Span attribute integration tests for the PgVector adapter.
//!
//! Skipped silently when `cognee_test_utils::pg_test_url()` returns `None`
//! (i.e. `DB_PROVIDER` is not set to `postgres`). Mirrors the gating
//! pattern in `pgvector_integration.rs`.
//!
//! Run with (note `--features pgvector`: without it the `cfg` below compiles the
//! whole file away and libtest reports a green `running 0 tests`):
//!
//!   DB_PROVIDER=postgres DB_HOST=localhost DB_PORT=5432 \
//!   DB_NAME=cognee_test DB_USERNAME=postgres DB_PASSWORD=postgres \
//!     cargo test -p cognee-vector --features pgvector,testing \
//!       --test pgvector_span_instrumentation -- --nocapture
#![cfg(feature = "pgvector")]

use cognee_test_utils::SpanCapture;
use cognee_vector::{PgVectorAdapter, VectorDB, VectorPoint};
use serial_test::serial;
use std::collections::HashMap;
use uuid::Uuid;

/// Dimension of the `DocumentChunk_text` collection every case below writes to.
/// Matches the 4-element vectors the tests build.
const DIM: usize = 4;

/// Returns a fresh adapter (with stale collections cleaned up, and the
/// collection the tests write to created) when a Postgres URL is configured, or
/// `None` to silently skip.
async fn make_adapter() -> Option<PgVectorAdapter> {
    let url = cognee_test_utils::pg_test_url()?;
    // `expect`, not `.ok()?`: past this point the URL *is* configured, so a
    // connect/migrate failure is a real defect. Collapsing it into `None` would
    // print the "DB_PROVIDER not set" skip line below and send whoever reads the
    // CI log hunting a broken service container instead of the adapter bug — and
    // it would make the guard's own claim that "a broken adapter would panic with
    // its own error instead" untrue. Same reasoning, same fix as
    // `crates/graph/tests/pg_graph_integration.rs`.
    let adapter = PgVectorAdapter::new(&url, DIM)
        .await
        .expect("DB_PROVIDER=postgres, so the adapter must connect and migrate");
    let cols = adapter
        .list_collections()
        .await
        .expect("listing collections must succeed against a live Postgres");
    for (dt, fname) in cols {
        adapter
            .delete_collection(&dt, &fname)
            .await
            .expect("dropping a stale collection must succeed");
    }
    // `index_points` deliberately does NOT create the collection: every
    // `VectorDB` impl in the workspace requires it to exist first (the
    // in-memory adapter returns `CollectionNotFound`, LanceDB fails to open the
    // table, pgvector's INSERT hits a missing relation), and the production
    // callers in `cognee-cognify` all do `has_collection` → `create_collection`
    // before indexing. Python's `create_data_points` auto-creates instead, but
    // that is a difference in where the responsibility sits, not in observable
    // pipeline behaviour. So the precondition is the test's job to establish —
    // omitting it is why all three cases here failed the first time they ever
    // executed with `--features pgvector` (see the module doc above).
    //
    // `expect` rather than `.ok()?`: a create failure must fail the test, not
    // silently degrade into the skip path this lane exists to detect.
    adapter
        .create_collection("DocumentChunk", "text", DIM)
        .await
        .expect("create DocumentChunk_text collection");
    Some(adapter)
}

#[tokio::test]
#[serial]
async fn upsert_emits_pgvector_span() {
    let Some(adapter) = make_adapter().await else {
        eprintln!("DB_PROVIDER not set to postgres — skipping upsert_emits_pgvector_span");
        return;
    };
    let capture = SpanCapture::install();

    let points: Vec<VectorPoint> = (0..2)
        .map(|i| VectorPoint {
            id: Uuid::new_v4(),
            vector: vec![i as f32, 0.0, 0.0, 0.0],
            metadata: HashMap::new(),
        })
        .collect();
    adapter
        .index_points("DocumentChunk", "text", &points)
        .await
        .expect("upsert");

    let spans = capture.spans();
    let s = spans
        .iter()
        .find(|s| s.name == "cognee.db.vector.upsert")
        .expect("expected upsert span");
    assert_eq!(s.field_str("cognee.db.system").as_deref(), Some("pgvector"));
    assert_eq!(
        s.field_str("cognee.vector.collection").as_deref(),
        Some("DocumentChunk_text"),
    );
    assert_eq!(s.field_i64("cognee.db.row_count"), Some(2));
}

#[tokio::test]
#[serial]
async fn search_emits_pgvector_span() {
    let Some(adapter) = make_adapter().await else {
        eprintln!("DB_PROVIDER not set to postgres — skipping search_emits_pgvector_span");
        return;
    };
    let capture = SpanCapture::install();

    let pid = Uuid::new_v4();
    adapter
        .index_points(
            "DocumentChunk",
            "text",
            &[VectorPoint {
                id: pid,
                vector: vec![0.1, 0.2, 0.3, 0.4],
                metadata: HashMap::new(),
            }],
        )
        .await
        .expect("seed");

    let results = adapter
        .search_similar("DocumentChunk", "text", &[0.1, 0.2, 0.3, 0.4], 5)
        .await
        .expect("search");
    assert!(!results.is_empty());

    let spans = capture.spans();
    let s = spans
        .iter()
        .find(|s| s.name == "cognee.db.vector.search")
        .expect("expected search span");
    assert_eq!(s.field_str("cognee.db.system").as_deref(), Some("pgvector"));
    assert_eq!(
        s.field_str("cognee.vector.collection").as_deref(),
        Some("DocumentChunk_text"),
    );
    assert!(s.field_i64("cognee.vector.result_count").unwrap_or(0) >= 1);
    // One row against top_k = 5: whatever plan answered, it was short, so the
    // exact fallback answered and the span must say so — that count is how a
    // full-scan latency regression gets attributed.
    assert_eq!(s.field_i64("cognee.vector.exact_fallbacks"), Some(1));

    // A full answer records an explicit zero, so "no fallback" is
    // distinguishable from "not instrumented".
    let capture = SpanCapture::install();
    adapter
        .search_similar("DocumentChunk", "text", &[0.1, 0.2, 0.3, 0.4], 1)
        .await
        .expect("search");
    let spans = capture.spans();
    let s = spans
        .iter()
        .find(|s| s.name == "cognee.db.vector.search")
        .expect("expected search span");
    assert_eq!(s.field_i64("cognee.vector.exact_fallbacks"), Some(0));
}

#[tokio::test]
#[serial]
async fn delete_emits_pgvector_span() {
    let Some(adapter) = make_adapter().await else {
        eprintln!("DB_PROVIDER not set to postgres — skipping delete_emits_pgvector_span");
        return;
    };
    let capture = SpanCapture::install();

    let pid = Uuid::new_v4();
    adapter
        .index_points(
            "DocumentChunk",
            "text",
            &[VectorPoint {
                id: pid,
                vector: vec![0.1, 0.0, 0.0, 0.0],
                metadata: HashMap::new(),
            }],
        )
        .await
        .expect("seed");
    adapter
        .delete_points("DocumentChunk", "text", &[pid])
        .await
        .expect("delete");

    let spans = capture.spans();
    let s = spans
        .iter()
        .find(|s| s.name == "cognee.db.vector.delete")
        .expect("expected delete span");
    assert_eq!(s.field_str("cognee.db.system").as_deref(), Some("pgvector"));
}
