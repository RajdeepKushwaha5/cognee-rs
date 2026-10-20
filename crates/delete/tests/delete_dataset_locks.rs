#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! `DeleteService::with_dataset_locks`: every dataset-scoped delete takes the
//! dataset's lock from the shared registry (Python's `dataset_lock`,
//! `datasets.py:198, 244`, `forget.py:298, 399@37f788bd2`), re-entrantly for a
//! caller that already holds it through `DatasetLocks::hold`, and not at all
//! without the hook. Plus `execute_datasets_by_id` skipping a dataset that
//! vanished before its turn.

use std::sync::Arc;
use std::time::Duration;

use cognee_core::dataset_locks::DatasetLocks;
use cognee_database::{self as database, DatabaseConnection, DeleteDb};
use cognee_delete::{DeleteError, DeleteMode, DeleteRequest, DeleteScope, DeleteService};
use cognee_models::{Data, Dataset};
use cognee_storage::{MockStorage, StorageTrait};
use cognee_test_utils::FailingDeleteDb;
use uuid::Uuid;

/// How long a delete must stay blocked to count as "waiting".
const BLOCKED: Duration = Duration::from_millis(150);
/// Upper bound for anything that must complete (a deadlock fails, not hangs).
const DONE: Duration = Duration::from_secs(10);

struct Fx {
    db: Arc<DatabaseConnection>,
    storage: Arc<MockStorage>,
    locks: Arc<DatasetLocks>,
    owner: Uuid,
}

async fn fixture() -> Fx {
    let db = database::connect("sqlite::memory:").await.unwrap();
    database::initialize(&db).await.unwrap();
    Fx {
        db: Arc::new(db),
        storage: Arc::new(MockStorage::new()),
        locks: Arc::new(DatasetLocks::new()),
        owner: Uuid::new_v4(),
    }
}

impl Fx {
    async fn dataset(&self, name: &str) -> Uuid {
        let dataset = Dataset::new(name.to_string(), self.owner, None, Uuid::new_v4());
        let id = dataset.id;
        database::ops::datasets::create_dataset(&self.db, dataset)
            .await
            .unwrap();
        id
    }

    async fn item(&self, dataset: Uuid) -> Uuid {
        let location = self.storage.store(b"content", "item.txt").await.unwrap();
        let data_id = Uuid::new_v4();
        let data = Data::builder(
            data_id,
            "item.txt",
            &location,
            "file://item.txt",
            "txt",
            "text/plain",
            format!("hash-{data_id}"),
            self.owner,
        )
        .build();
        database::ops::data::create_data(&self.db, data)
            .await
            .unwrap();
        database::ops::datasets::attach_data_to_dataset(&self.db, dataset, data_id)
            .await
            .unwrap();
        data_id
    }

    fn plain_over(&self, db: Arc<dyn DeleteDb>) -> DeleteService {
        DeleteService::new(self.storage.clone() as Arc<dyn StorageTrait>, db)
    }

    fn plain(&self) -> DeleteService {
        self.plain_over(self.db.clone() as Arc<dyn DeleteDb>)
    }

    fn hooked(&self) -> DeleteService {
        self.plain().with_dataset_locks(Arc::clone(&self.locks))
    }

    async fn exists(&self, id: Uuid) -> bool {
        DeleteDb::get_dataset_by_id(self.db.as_ref(), id)
            .await
            .unwrap()
            .is_some()
    }
}

fn req(scope: DeleteScope) -> DeleteRequest {
    DeleteRequest {
        scope,
        mode: DeleteMode::Soft,
        memory_only: false,
    }
}

/// Scenario: a pipeline holds dataset `held`'s lock; a hooked delete of
/// `scope` (which touches `held`) starts.
/// Expected: the delete waits — it has not finished and `held` still exists
/// while the lock is held — and completes once the lock is released.
/// Verification: spawn the delete, check it is still pending after `BLOCKED`,
/// release, then await it under `DONE`.
async fn assert_waits_for(f: &Fx, held: Uuid, scope: DeleteScope, what: &str) {
    let guard = f.locks.lock(held).await;
    let svc = Arc::new(f.hooked());
    let task = {
        let svc = Arc::clone(&svc);
        tokio::spawn(async move { svc.execute(&req(scope)).await })
    };
    tokio::time::sleep(BLOCKED).await;
    assert!(!task.is_finished(), "{what}: delete must wait for the lock");
    drop(guard);
    tokio::time::timeout(DONE, task)
        .await
        .unwrap_or_else(|_| panic!("{what}: delete must finish after the release"))
        .unwrap()
        .unwrap_or_else(|e| panic!("{what}: delete failed: {e}"));
}

#[tokio::test]
async fn every_dataset_scoped_delete_waits_for_a_held_lock() {
    let f = fixture().await;

    let ds = f.dataset("by-id").await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::DatasetById { dataset_id: ds },
        "DatasetById",
    )
    .await;
    assert!(!f.exists(ds).await);

    let ds = f.dataset("item-in").await;
    let item = f.item(ds).await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::DataInDataset {
            dataset_id: ds,
            data_id: item,
            delete_dataset_if_empty: false,
        },
        "DataInDataset",
    )
    .await;

    let ds = f.dataset("by-name").await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::Dataset {
            owner_id: f.owner,
            dataset_name: "by-name".into(),
            tenant_id: None,
        },
        "Dataset by name",
    )
    .await;
    assert!(!f.exists(ds).await);

    let ds = f.dataset("item-by-name").await;
    let item = f.item(ds).await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::Data {
            owner_id: f.owner,
            data_id: item,
            dataset_name: Some("item-by-name".into()),
            delete_dataset_if_empty: false,
            tenant_id: None,
        },
        "Data with a dataset name",
    )
    .await;

    let ds = f.dataset("item-anywhere").await;
    let item = f.item(ds).await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::Data {
            owner_id: f.owner,
            data_id: item,
            dataset_name: None,
            delete_dataset_if_empty: false,
            tenant_id: None,
        },
        "Data without a dataset name",
    )
    .await;

    let ds = f.dataset("user-scope").await;
    assert_waits_for(
        &f,
        ds,
        DeleteScope::User {
            owner_id: f.owner,
            tenant_id: None,
        },
        "User",
    )
    .await;
    assert!(!f.exists(ds).await);
}

/// Scenario: a pipeline holds the dataset's lock; a delete built **without**
/// the hook runs.
/// Expected: today's behaviour — it takes no lock and completes at once.
#[tokio::test]
async fn no_lock_is_taken_without_the_hook() {
    let f = fixture().await;
    let ds = f.dataset("unhooked").await;
    let _guard = f.locks.lock(ds).await;
    tokio::time::timeout(
        DONE,
        f.plain()
            .execute(&req(DeleteScope::DatasetById { dataset_id: ds })),
    )
    .await
    .expect("an unhooked delete must not wait on the registry")
    .unwrap();
    assert!(!f.exists(ds).await);
}

/// Scenario: an operation holds the dataset through `DatasetLocks::hold` and,
/// inside it, deletes an item with the hooked service and then locks the
/// dataset again the way an `add` does (`DatasetLocks::lock`).
/// Expected: no deadlock — both inner acquisitions are re-entrant — while the
/// dataset stays excluded for other tasks.
#[tokio::test]
async fn a_caller_already_holding_the_dataset_does_not_deadlock() {
    let f = fixture().await;
    let ds = f.dataset("update-like").await;
    let item = f.item(ds).await;
    let svc = f.hooked();
    tokio::time::timeout(
        DONE,
        f.locks.hold([ds], async {
            svc.execute(&req(DeleteScope::DataInDataset {
                dataset_id: ds,
                data_id: item,
                delete_dataset_if_empty: false,
            }))
            .await
            .unwrap();
            let readd = f.locks.lock(ds).await;
            assert!(readd.is_reentrant());
        }),
    )
    .await
    .expect("a re-entrant delete must not deadlock its caller");
}

/// Scenario (cloud finding #1): `execute_datasets_by_id([a, ghost, b])` where
/// `ghost` never existed (or was deleted after the caller listed it).
/// Expected: `Ok`, the two real datasets deleted, the ghost skipped.
#[tokio::test]
async fn execute_datasets_by_id_skips_a_vanished_dataset() {
    let f = fixture().await;
    let a = f.dataset("a").await;
    let b = f.dataset("b").await;
    let ghost = Uuid::new_v4();
    for svc in [f.plain(), f.hooked()] {
        let a = f.dataset("a2").await;
        let b = f.dataset("b2").await;
        let result = svc
            .execute_datasets_by_id(&[a, ghost, b], DeleteMode::Soft, false)
            .await
            .unwrap();
        assert_eq!(result.deleted_datasets, 2);
        assert!(!f.exists(a).await && !f.exists(b).await);
    }
    assert!(f.exists(a).await && f.exists(b).await, "untouched");
}

/// Scenario: the realistic race — the loop reaches `b` while another
/// operation holds `b`'s lock, and that operation deletes `b` before letting
/// go.
/// Expected: the hooked loop waits, re-reads `b` under its lock, finds it gone
/// and skips it: `Ok`, `a` and `c` deleted.
#[tokio::test]
async fn a_dataset_deleted_while_the_loop_waits_for_its_lock_is_skipped() {
    let f = fixture().await;
    let (a, b, c) = (
        f.dataset("a").await,
        f.dataset("b").await,
        f.dataset("c").await,
    );
    let guard = f.locks.lock(b).await;
    let svc = Arc::new(f.hooked());
    let task = {
        let svc = Arc::clone(&svc);
        tokio::spawn(async move {
            svc.execute_datasets_by_id(&[a, b, c], DeleteMode::Soft, false)
                .await
        })
    };
    tokio::time::sleep(BLOCKED).await;
    assert!(!task.is_finished(), "the loop must wait at b");
    assert!(!f.exists(a).await, "a was deleted before b's lock");
    database::ops::datasets::delete_dataset(&f.db, b)
        .await
        .unwrap();
    drop(guard);
    let result = tokio::time::timeout(DONE, task)
        .await
        .expect("the loop must finish after the release")
        .unwrap()
        .unwrap();
    assert_eq!(result.deleted_datasets, 2);
    assert!(!f.exists(c).await);
}

/// Scenario: a skip does not swallow a real failure — `[a, ghost, b]` where
/// deleting `b` fails, and `[ghost, b]` with the same failure.
/// Expected: the first is `Partial` (a completed, b failed); the second,
/// with nothing completed, returns the error as is.
#[tokio::test]
async fn a_real_failure_still_stops_the_loop() {
    let f = fixture().await;
    let a = f.dataset("a").await;
    let b = f.dataset("b").await;
    let ghost = Uuid::new_v4();
    let failing = || {
        f.plain_over(Arc::new(FailingDeleteDb::new(
            f.db.clone() as Arc<dyn DeleteDb>,
            Some(b),
        )))
        .with_dataset_locks(Arc::clone(&f.locks))
    };

    match failing()
        .execute_datasets_by_id(&[a, ghost, b], DeleteMode::Soft, false)
        .await
    {
        Err(DeleteError::Partial {
            completed,
            failed_dataset_id,
            ..
        }) => {
            assert_eq!(completed.deleted_datasets, 1);
            assert_eq!(failed_dataset_id, b);
        }
        other => panic!("expected Partial, got {other:?}"),
    }
    assert!(!f.exists(a).await);
    assert!(f.exists(b).await);

    let result = failing()
        .execute_datasets_by_id(&[ghost, b], DeleteMode::Soft, false)
        .await;
    assert!(
        matches!(result, Err(ref e) if !matches!(e, DeleteError::Partial { .. })),
        "nothing completed: the error as is, got {result:?}"
    );
}
