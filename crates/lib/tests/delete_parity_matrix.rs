#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Delete-parity matrix (SDK-936), library half: `DatasetManager` with and
//! without an ACL, and `forget`, × {grant (G), no grant (N), grant only in
//! another tenant (X), no ACL (A)}.
//!
//! Rules (Python @37f788bd2): under an ACL the grant counts only in the
//! caller's tenant (`get_all_user_permission_datasets.py:49-53`); without one
//! ownership in the tenant replaces it (`get_dataset_ids.py:29-34`).
//! `list_data`/`has_data` are always authorized (`datasets.py:155, 164`);
//! `delete_all` / `forget(everything)` are tenant-scoped, by id, and stop at
//! the first error (`datasets.py:316-323`, `forget.py:191-205`).
//!
//! Fixture as in `crates/delete/tests/delete_parity_matrix.rs`: owner O has
//! `"shared"` in T2, T1 and tenant-less (seeded in that order), one item
//! each, all four permissions; grantee G holds `delete` on the T1 and T2
//! rows; outsider N holds nothing.

use std::sync::Arc;

use cognee::api::{
    ApiError, DatasetDb, DatasetError, DatasetManager, DatasetRef, ForgetTarget, forget,
};
use cognee_database::{self as database, AclDb, DatabaseConnection, DeleteDb, IngestDb};
use cognee_delete::{DeleteError, DeleteMode, DeleteService};
use cognee_models::{Data, Dataset};
use cognee_storage::{MockStorage, StorageTrait};
use cognee_test_utils::{FailingDeleteDb, MockAclDb};
use uuid::Uuid;

const NAME: &str = "shared";

struct Fx {
    db: Arc<DatabaseConnection>,
    storage: Arc<MockStorage>,
    acl: Arc<MockAclDb>,
    owner: Uuid,
    grantee: Uuid,
    outsider: Uuid,
    t1: Uuid,
    t2: Uuid,
    o_t2: (Uuid, Uuid),
    o_t1: (Uuid, Uuid),
    o_none: (Uuid, Uuid),
}

async fn seed_dataset(
    db: &DatabaseConnection,
    acl: &MockAclDb,
    owner: Uuid,
    name: &str,
    tenant: Option<Uuid>,
) -> Uuid {
    let dataset = Dataset::new(name.to_string(), owner, tenant, Uuid::new_v4());
    let id = dataset.id;
    database::ops::datasets::create_dataset(db, dataset)
        .await
        .unwrap();
    if let Some(t) = tenant {
        acl.set_dataset_tenant(id, t);
    }
    id
}

async fn seed_item(
    db: &DatabaseConnection,
    storage: &MockStorage,
    owner: Uuid,
    dataset: Uuid,
) -> Uuid {
    let location = storage.store(b"content", "item.txt").await.unwrap();
    let data_id = Uuid::new_v4();
    let data = Data::builder(
        data_id,
        "item.txt",
        &location,
        "file://item.txt",
        "txt",
        "text/plain",
        format!("hash-{data_id}"),
        owner,
    )
    .build();
    database::ops::data::create_data(db, data).await.unwrap();
    database::ops::datasets::attach_data_to_dataset(db, dataset, data_id)
        .await
        .unwrap();
    data_id
}

async fn grant(acl: &MockAclDb, principal: Uuid, dataset: Uuid, permission: &str) {
    let acl: &dyn AclDb = acl;
    acl.ensure_principal(principal, "user").await.unwrap();
    acl.grant_permission(principal, dataset, permission)
        .await
        .unwrap();
}

async fn fixture() -> Fx {
    let db = database::connect("sqlite::memory:").await.unwrap();
    database::initialize(&db).await.unwrap();
    let db = Arc::new(db);
    let storage = Arc::new(MockStorage::new());
    let acl = Arc::new(MockAclDb::new());
    let (owner, grantee, outsider) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let (t1, t2) = (Uuid::new_v4(), Uuid::new_v4());
    let mut rows = Vec::new();
    for tenant in [Some(t2), Some(t1), None] {
        let ds = seed_dataset(&db, &acl, owner, NAME, tenant).await;
        let item = seed_item(&db, &storage, owner, ds).await;
        for p in ["read", "write", "delete", "share"] {
            grant(&acl, owner, ds, p).await;
        }
        rows.push((ds, item));
    }
    grant(&acl, grantee, rows[1].0, "delete").await;
    grant(&acl, grantee, rows[0].0, "delete").await;
    Fx {
        db,
        storage,
        acl,
        owner,
        grantee,
        outsider,
        t1,
        t2,
        o_t2: rows[0],
        o_t1: rows[1],
        o_none: rows[2],
    }
}

impl Fx {
    fn svc_over(&self, db: Arc<dyn DeleteDb>) -> DeleteService {
        DeleteService::new(self.storage.clone() as Arc<dyn StorageTrait>, db)
    }

    fn svc(&self) -> DeleteService {
        self.svc_over(self.db.clone() as Arc<dyn DeleteDb>)
    }

    fn with_acl(&self) -> DatasetManager {
        DatasetManager::new(self.db.clone() as Arc<dyn DatasetDb>)
            .with_acl(self.acl.clone() as Arc<dyn AclDb>)
    }

    fn without_acl(&self) -> DatasetManager {
        DatasetManager::new(self.db.clone() as Arc<dyn DatasetDb>)
    }

    async fn dataset_exists(&self, id: Uuid) -> bool {
        database::ops::datasets::get_dataset(&self.db, id)
            .await
            .unwrap()
            .is_some()
    }

    async fn data_exists(&self, id: Uuid) -> bool {
        database::ops::data::get_data(&self.db, id)
            .await
            .unwrap()
            .is_some()
    }

    async fn assert_untouched(&self, what: &str) {
        for (ds, item) in [self.o_t2, self.o_t1, self.o_none] {
            assert!(self.dataset_exists(ds).await, "{what}: dataset {ds} gone");
            assert!(self.data_exists(item).await, "{what}: item {item} gone");
        }
    }

    fn ingest(&self) -> &dyn IngestDb {
        self.db.as_ref()
    }
}

fn assert_denied<T: std::fmt::Debug>(result: Result<T, DatasetError>, what: &str) {
    assert!(
        matches!(result, Err(DatasetError::PermissionDenied)),
        "{what}: expected PermissionDenied, got {result:?}"
    );
}

/// The no-ACL "not found" of the delete paths: the ownership fallback.
fn assert_delete_not_found<T: std::fmt::Debug>(result: Result<T, DatasetError>, what: &str) {
    assert!(
        matches!(
            result,
            Err(DatasetError::Delete(DeleteError::Validation(ref m))) if m.contains("not found")
        ),
        "{what}: expected a not-found validation, got {result:?}"
    );
}

fn assert_not_found<T: std::fmt::Debug>(result: Result<T, DatasetError>, what: &str) {
    assert!(
        matches!(result, Err(DatasetError::NotFound)),
        "{what}: expected NotFound, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// empty_dataset
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manager_empty_dataset_grant_deletes_foreign_dataset() {
    let f = fixture().await;
    f.with_acl()
        .empty_dataset(f.o_t1.0, f.grantee, Some(f.t1), &f.svc())
        .await
        .expect("G holds delete on O's T1 dataset");
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
}

#[tokio::test]
async fn manager_empty_dataset_no_grant_denied() {
    let f = fixture().await;
    assert_denied(
        f.with_acl()
            .empty_dataset(f.o_t1.0, f.outsider, Some(f.t1), &f.svc())
            .await,
        "N",
    );
    f.assert_untouched("N").await;
}

#[tokio::test]
async fn manager_empty_dataset_other_tenant_grant_denied() {
    let f = fixture().await;
    assert_denied(
        f.with_acl()
            .empty_dataset(f.o_t2.0, f.grantee, Some(f.t1), &f.svc())
            .await,
        "X",
    );
    f.assert_untouched("X").await;
}

#[tokio::test]
async fn manager_empty_dataset_without_acl_requires_owner_and_tenant() {
    let f = fixture().await;
    let m = f.without_acl();
    assert_delete_not_found(
        m.empty_dataset(f.o_t1.0, f.grantee, Some(f.t1), &f.svc())
            .await,
        "A: not the owner",
    );
    assert_delete_not_found(
        m.empty_dataset(f.o_t1.0, f.owner, Some(f.t2), &f.svc())
            .await,
        "A: owner in another tenant",
    );
    f.assert_untouched("A").await;
    m.empty_dataset(f.o_t1.0, f.owner, Some(f.t1), &f.svc())
        .await
        .unwrap();
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);
}

// ---------------------------------------------------------------------------
// delete_data
// ---------------------------------------------------------------------------

async fn delete_item(
    f: &Fx,
    m: &DatasetManager,
    row: (Uuid, Uuid),
    caller: Uuid,
    tenant: Option<Uuid>,
) -> Result<cognee_delete::DeleteResult, DatasetError> {
    m.delete_data(
        row.0,
        row.1,
        caller,
        tenant,
        DeleteMode::Soft,
        false,
        &f.svc(),
    )
    .await
}

#[tokio::test]
async fn manager_delete_data_grant_deletes() {
    let f = fixture().await;
    delete_item(&f, &f.with_acl(), f.o_t1, f.grantee, Some(f.t1))
        .await
        .unwrap();
    assert!(!f.data_exists(f.o_t1.1).await);
    assert!(f.data_exists(f.o_t2.1).await);
}

#[tokio::test]
async fn manager_delete_data_no_grant_denied() {
    let f = fixture().await;
    assert_denied(
        delete_item(&f, &f.with_acl(), f.o_t1, f.outsider, Some(f.t1)).await,
        "N",
    );
    f.assert_untouched("N").await;
}

#[tokio::test]
async fn manager_delete_data_other_tenant_grant_denied() {
    let f = fixture().await;
    assert_denied(
        delete_item(&f, &f.with_acl(), f.o_t2, f.grantee, Some(f.t1)).await,
        "X",
    );
    f.assert_untouched("X").await;
}

#[tokio::test]
async fn manager_delete_data_without_acl_requires_owner_and_tenant() {
    let f = fixture().await;
    let m = f.without_acl();
    assert_delete_not_found(
        delete_item(&f, &m, f.o_t1, f.grantee, Some(f.t1)).await,
        "A: not the owner",
    );
    assert_delete_not_found(
        delete_item(&f, &m, f.o_t1, f.owner, None).await,
        "A: owner, tenant-less caller",
    );
    f.assert_untouched("A").await;
    delete_item(&f, &m, f.o_t1, f.owner, Some(f.t1))
        .await
        .unwrap();
    assert!(!f.data_exists(f.o_t1.1).await);
    assert!(f.data_exists(f.o_none.1).await);
}

// ---------------------------------------------------------------------------
// delete_all
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manager_delete_all_deletes_delete_authorized_in_tenant() {
    let f = fixture().await;
    let results = f
        .with_acl()
        .delete_all(f.grantee, Some(f.t1), &f.svc())
        .await
        .unwrap();
    assert_eq!(results.len(), 1);
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);
}

#[tokio::test]
async fn manager_delete_all_no_grants_deletes_nothing() {
    let f = fixture().await;
    let results = f
        .with_acl()
        .delete_all(f.outsider, Some(f.t1), &f.svc())
        .await
        .unwrap();
    assert!(results.is_empty());
    f.assert_untouched("N").await;
}

#[tokio::test]
async fn manager_delete_all_ignores_other_tenant_grants() {
    let f = fixture().await;
    f.with_acl()
        .delete_all(f.grantee, Some(f.t2), &f.svc())
        .await
        .unwrap();
    assert!(!f.dataset_exists(f.o_t2.0).await);
    assert!(
        f.dataset_exists(f.o_t1.0).await,
        "the T1 grant is not used from T2"
    );
    assert!(f.dataset_exists(f.o_none.0).await);
}

#[tokio::test]
async fn manager_delete_all_without_acl_is_owner_and_tenant() {
    let f = fixture().await;
    let none = f
        .without_acl()
        .delete_all(f.grantee, Some(f.t1), &f.svc())
        .await
        .unwrap();
    assert!(none.is_empty(), "G owns nothing");
    f.assert_untouched("A: G").await;
    f.without_acl()
        .delete_all(f.owner, Some(f.t1), &f.svc())
        .await
        .unwrap();
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);
}

// ---------------------------------------------------------------------------
// list_data / has_data
// ---------------------------------------------------------------------------

#[tokio::test]
async fn manager_reads_with_read_grant() {
    let f = fixture().await;
    grant(&f.acl, f.grantee, f.o_t1.0, "read").await;
    let m = f.with_acl();
    let items = m.list_data(f.o_t1.0, f.grantee, Some(f.t1)).await.unwrap();
    assert_eq!(items.len(), 1);
    assert!(m.has_data(f.o_t1.0, f.grantee, Some(f.t1)).await.unwrap());
}

#[tokio::test]
async fn manager_reads_without_grant_denied() {
    let f = fixture().await;
    let m = f.with_acl();
    // G holds `delete`, not `read`, on o_t1; N holds nothing.
    for caller in [f.grantee, f.outsider] {
        assert_denied(m.list_data(f.o_t1.0, caller, Some(f.t1)).await, "N list");
        assert_denied(m.has_data(f.o_t1.0, caller, Some(f.t1)).await, "N has");
    }
}

#[tokio::test]
async fn manager_reads_other_tenant_grant_denied() {
    let f = fixture().await;
    grant(&f.acl, f.grantee, f.o_t2.0, "read").await;
    let m = f.with_acl();
    assert_denied(m.list_data(f.o_t2.0, f.grantee, Some(f.t1)).await, "X list");
    assert_denied(m.has_data(f.o_t2.0, f.grantee, Some(f.t1)).await, "X has");
}

#[tokio::test]
async fn manager_reads_without_acl_require_owner_and_tenant() {
    let f = fixture().await;
    let m = f.without_acl();
    assert_eq!(
        m.list_data(f.o_t1.0, f.owner, Some(f.t1))
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(m.has_data(f.o_t1.0, f.owner, Some(f.t1)).await.unwrap());
    for (dataset, caller, tenant, what) in [
        (f.o_t1.0, f.grantee, Some(f.t1), "not the owner"),
        (f.o_t1.0, f.owner, None, "owner, other tenant"),
        (Uuid::new_v4(), f.owner, Some(f.t1), "unknown id"),
    ] {
        assert_not_found(m.list_data(dataset, caller, tenant).await, what);
        assert_not_found(m.has_data(dataset, caller, tenant).await, what);
    }
}

// ---------------------------------------------------------------------------
// forget (no ACL by construction: the A column)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn forget_dataset_by_name_and_id_owner_and_tenant_only() {
    let f = fixture().await;
    let svc = f.svc();
    forget(
        ForgetTarget::Dataset {
            dataset: DatasetRef::Name(NAME.to_string()),
        },
        f.owner,
        Some(f.t1),
        &svc,
        Some(f.ingest()),
    )
    .await
    .unwrap();
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);

    // By id: must be the owner's and in the caller's tenant.
    for (owner, tenant, what) in [
        (f.owner, Some(f.t1), "owner, other tenant"),
        (f.grantee, Some(f.t2), "not the owner"),
    ] {
        let result = forget(
            ForgetTarget::Dataset {
                dataset: DatasetRef::Id(f.o_t2.0),
            },
            owner,
            tenant,
            &svc,
            Some(f.ingest()),
        )
        .await;
        assert!(
            matches!(result, Err(ApiError::InvalidArgument(_))),
            "{what}: got {result:?}"
        );
    }
    assert!(f.dataset_exists(f.o_t2.0).await);
    forget(
        ForgetTarget::Dataset {
            dataset: DatasetRef::Id(f.o_t2.0),
        },
        f.owner,
        Some(f.t2),
        &svc,
        Some(f.ingest()),
    )
    .await
    .unwrap();
    assert!(!f.dataset_exists(f.o_t2.0).await);
}

#[tokio::test]
async fn forget_item_owner_and_tenant_only() {
    let f = fixture().await;
    let svc = f.svc();
    // From no tenant, "shared" is the tenant-less row, which does not hold
    // o_t1's item: refused, the item stays.
    let result = forget(
        ForgetTarget::Item {
            data_id: f.o_t1.1,
            dataset: DatasetRef::Name(NAME.to_string()),
        },
        f.owner,
        None,
        &svc,
        Some(f.ingest()),
    )
    .await;
    assert!(result.is_err(), "got {result:?}");
    f.assert_untouched("forget item, wrong tenant").await;

    forget(
        ForgetTarget::Item {
            data_id: f.o_t1.1,
            dataset: DatasetRef::Name(NAME.to_string()),
        },
        f.owner,
        Some(f.t1),
        &svc,
        Some(f.ingest()),
    )
    .await
    .unwrap();
    assert!(!f.data_exists(f.o_t1.1).await);
    assert!(f.data_exists(f.o_t2.1).await);
    assert!(f.data_exists(f.o_none.1).await);
}

#[tokio::test]
async fn forget_all_is_tenant_scoped_and_by_id() {
    let f = fixture().await;
    let svc = f.svc();
    let result = forget(ForgetTarget::All, f.owner, Some(f.t1), &svc, None)
        .await
        .unwrap();
    assert_eq!(result.delete_result.deleted_datasets, 1);
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);

    forget(ForgetTarget::All, f.owner, None, &svc, None)
        .await
        .unwrap();
    assert!(!f.dataset_exists(f.o_none.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await, "T2 never touched");
}

#[tokio::test]
async fn forget_all_stops_at_first_failure_with_partial() {
    let f = fixture().await;
    let p = Uuid::new_v4();
    for name in ["p1", "p2"] {
        let ds = seed_dataset(&f.db, &f.acl, p, name, Some(f.t1)).await;
        seed_item(&f.db, &f.storage, p, ds).await;
    }
    let order: Vec<Uuid> = DeleteDb::list_datasets_by_owner(f.db.as_ref(), p)
        .await
        .unwrap()
        .into_iter()
        .map(|ds| ds.id)
        .collect();
    let svc = f.svc_over(Arc::new(FailingDeleteDb::new(
        f.db.clone() as Arc<dyn DeleteDb>,
        Some(order[1]),
    )));
    let result = forget(ForgetTarget::All, p, Some(f.t1), &svc, None).await;
    match result {
        Err(ApiError::DeleteErr(DeleteError::Partial {
            completed,
            failed_dataset_id,
            ..
        })) => {
            assert_eq!(completed.deleted_datasets, 1);
            assert_eq!(failed_dataset_id, order[1]);
        }
        other => panic!("expected DeleteErr(Partial), got {other:?}"),
    }
    assert!(!f.dataset_exists(order[0]).await);
    assert!(f.dataset_exists(order[1]).await);
    f.assert_untouched("other owner").await;
}
