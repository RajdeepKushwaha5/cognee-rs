#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Delete-parity matrix (SDK-936), CLI part: the delete request
//! builder, run against the service it is built for.
//!
//! - `RequestAuthorization::Grant` (the cloud `delete --enforce-acl`): a
//!   grantee reaches another user's dataset **by id** when it holds `delete`
//!   in its tenant; names stay the caller's own; a dataset-less item delete is
//!   refused.
//! - `RequestAuthorization::Ownership` (OSS, no ACL): every reference must be
//!   the caller's own dataset in its tenant and is resolved to an id; `--all`
//!   is the caller's datasets in the tenant (Python `datasets.delete_all`).
//!
//! Fixture: owner O has three datasets named `"shared"` — `o_t2` (T2),
//! `o_t1` (T1), `o_none` (tenant-less), seeded in that order so a
//! tenant-loose lookup would hit the wrong row first. G holds `delete` on
//! `o_t1` and `o_t2`; N holds nothing.

use std::sync::Arc;

use clap::Parser;
use cognee::database::{self as database, AclDb, DatabaseConnection, DeleteDb, IngestDb};
use cognee::delete::{
    AuthorizedDeleteService, DeleteError, DeleteRequest, DeleteScope, DeleteService,
};
use cognee::models::{Data, Dataset};
use cognee_cli::cli::{Cli, Commands, DeleteArgs};
use cognee_cli::commands::delete::{
    RequestAuthorization, build_request_async, resolve_caller, validate_scope_selection,
};
use cognee_cli::error::CliError;
use cognee_storage::{MockStorage, StorageTrait};
use cognee_test_utils::MockAclDb;
use uuid::Uuid;

struct Fixture {
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

async fn seed(
    db: &DatabaseConnection,
    storage: &MockStorage,
    acl: &MockAclDb,
    owner: Uuid,
    tenant: Option<Uuid>,
) -> (Uuid, Uuid) {
    let dataset = Dataset::new("shared".to_string(), owner, tenant, Uuid::new_v4());
    let dataset_id = dataset.id;
    database::ops::datasets::create_dataset(db, dataset)
        .await
        .unwrap();
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
    database::ops::datasets::attach_data_to_dataset(db, dataset_id, data_id)
        .await
        .unwrap();
    if let Some(t) = tenant {
        acl.set_dataset_tenant(dataset_id, t);
    }
    (dataset_id, data_id)
}

async fn grant(acl: &MockAclDb, principal: Uuid, dataset: Uuid, permission: &str) {
    let acl: &dyn AclDb = acl;
    acl.ensure_principal(principal, "user").await.unwrap();
    acl.grant_permission(principal, dataset, permission)
        .await
        .unwrap();
}

async fn fixture() -> Fixture {
    let db = database::connect("sqlite::memory:").await.unwrap();
    database::initialize(&db).await.unwrap();
    let db = Arc::new(db);
    let storage = Arc::new(MockStorage::new());
    let acl = Arc::new(MockAclDb::new());
    let (owner, grantee, outsider) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let (t1, t2) = (Uuid::new_v4(), Uuid::new_v4());
    let o_t2 = seed(&db, &storage, &acl, owner, Some(t2)).await;
    let o_t1 = seed(&db, &storage, &acl, owner, Some(t1)).await;
    let o_none = seed(&db, &storage, &acl, owner, None).await;
    for ds in [o_t2.0, o_t1.0, o_none.0] {
        for p in ["read", "write", "delete", "share"] {
            grant(&acl, owner, ds, p).await;
        }
    }
    grant(&acl, grantee, o_t1.0, "delete").await;
    grant(&acl, grantee, o_t2.0, "delete").await;
    Fixture {
        db,
        storage,
        acl,
        owner,
        grantee,
        outsider,
        t1,
        t2,
        o_t2,
        o_t1,
        o_none,
    }
}

impl Fixture {
    fn plain(&self) -> DeleteService {
        DeleteService::new(
            self.storage.clone() as Arc<dyn StorageTrait>,
            self.db.clone() as Arc<dyn DeleteDb>,
        )
    }

    fn authorized(&self) -> AuthorizedDeleteService {
        AuthorizedDeleteService::new(
            self.plain(),
            self.acl.clone() as Arc<dyn AclDb>,
            self.db.clone() as Arc<dyn DeleteDb>,
        )
    }

    async fn build(
        &self,
        argv: &[&str],
        authorization: RequestAuthorization,
    ) -> Result<(DeleteRequest, Uuid, Option<Uuid>), CliError> {
        let args = delete_args(argv);
        let (caller, tenant) = resolve_caller(&args, &self.owner.to_string())?;
        let request = build_request_async(
            args,
            caller,
            tenant,
            self.db.clone() as Arc<dyn IngestDb>,
            authorization,
        )
        .await?;
        Ok((request, caller, tenant))
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
}

fn delete_args(argv: &[&str]) -> DeleteArgs {
    let mut full = vec!["cognee-cli", "delete"];
    full.extend_from_slice(argv);
    match Cli::try_parse_from(full).expect("argv parses").command {
        Commands::Delete(args) => args,
        other => panic!("expected delete, got {other:?}"),
    }
}

fn assert_denied<T: std::fmt::Debug>(result: Result<T, DeleteError>, what: &str) {
    match result {
        Err(DeleteError::PermissionDenied(_)) => {}
        other => panic!("{what}: expected PermissionDenied, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Grant
// ---------------------------------------------------------------------------

#[tokio::test]
async fn grant_builder_dataset_id_lets_grantee_delete() {
    let f = fixture().await;
    let (g, t1, d) = (
        f.grantee.to_string(),
        f.t1.to_string(),
        f.o_t1.0.to_string(),
    );
    let (request, caller, tenant) = f
        .build(
            &[
                "--user-id",
                &g,
                "--tenant-id",
                &t1,
                "--dataset-id",
                &d,
                "--force",
            ],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert!(
        matches!(request.scope, DeleteScope::DatasetById { dataset_id } if dataset_id == f.o_t1.0)
    );
    assert_eq!((caller, tenant), (f.grantee, Some(f.t1)));
    f.authorized()
        .execute(&request, caller, tenant)
        .await
        .expect("a delete grant in the caller's tenant admits the grantee");
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
    assert!(f.dataset_exists(f.o_none.0).await);
}

#[tokio::test]
async fn grant_builder_dataset_id_no_grant_denied() {
    let f = fixture().await;
    let (n, t1, d) = (
        f.outsider.to_string(),
        f.t1.to_string(),
        f.o_t1.0.to_string(),
    );
    let (request, caller, tenant) = f
        .build(
            &["--user-id", &n, "--tenant-id", &t1, "--dataset-id", &d],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert_denied(
        f.authorized().execute(&request, caller, tenant).await,
        "no grant",
    );
    assert!(f.data_exists(f.o_t1.1).await);
}

#[tokio::test]
async fn grant_builder_dataset_id_other_tenant_denied() {
    let f = fixture().await;
    // G holds `delete` on o_t2, but acts in T1.
    let (g, t1, d) = (
        f.grantee.to_string(),
        f.t1.to_string(),
        f.o_t2.0.to_string(),
    );
    let (request, caller, tenant) = f
        .build(
            &["--user-id", &g, "--tenant-id", &t1, "--dataset-id", &d],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert_denied(
        f.authorized().execute(&request, caller, tenant).await,
        "grant only in another tenant",
    );
    assert!(f.data_exists(f.o_t2.1).await);
}

#[tokio::test]
async fn grant_builder_dataset_name_is_owner_scoped() {
    let f = fixture().await;
    // G names "shared": a name is the caller's own, so it does not reach O's
    // dataset even though G holds `delete` on it.
    let (g, t1) = (f.grantee.to_string(), f.t1.to_string());
    let (request, caller, tenant) = f
        .build(
            &[
                "--user-id",
                &g,
                "--tenant-id",
                &t1,
                "--dataset-name",
                "shared",
            ],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert!(matches!(
        &request.scope,
        DeleteScope::Dataset { owner_id, dataset_name, tenant_id }
            if *owner_id == f.grantee && dataset_name == "shared" && *tenant_id == Some(f.t1)
    ));
    assert!(matches!(
        f.authorized().execute(&request, caller, tenant).await,
        Err(DeleteError::Validation(_))
    ));
    assert!(f.dataset_exists(f.o_t1.0).await);

    // With a data id and a name, the same owner scoping holds.
    let x = f.o_t1.1.to_string();
    let (request, ..) = f
        .build(
            &[
                "--user-id",
                &g,
                "--tenant-id",
                &t1,
                "--data-id",
                &x,
                "--dataset-name",
                "shared",
            ],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert!(matches!(
        &request.scope,
        DeleteScope::Data { owner_id, dataset_name: Some(_), tenant_id, .. }
            if *owner_id == f.grantee && *tenant_id == Some(f.t1)
    ));
}

#[tokio::test]
async fn grant_builder_rejects_dataset_less_item() {
    let f = fixture().await;
    let x = f.o_t1.1.to_string();
    let result = f
        .build(&["--data-id", &x], RequestAuthorization::Grant)
        .await;
    assert!(
        matches!(result, Err(CliError::Validation(ref m)) if m.contains("requires --dataset-name or --dataset-id")),
        "got {result:?}"
    );
}

#[tokio::test]
async fn grant_builder_all_is_the_service_expansion() {
    let f = fixture().await;
    let (g, t1) = (f.grantee.to_string(), f.t1.to_string());
    let (request, caller, tenant) = f
        .build(
            &["--user-id", &g, "--tenant-id", &t1, "--all"],
            RequestAuthorization::Grant,
        )
        .await
        .unwrap();
    assert!(matches!(request.scope, DeleteScope::All));
    f.authorized()
        .execute(&request, caller, tenant)
        .await
        .unwrap();
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await, "other tenant untouched");
    assert!(f.dataset_exists(f.o_none.0).await, "ungranted untouched");
}

// ---------------------------------------------------------------------------
// Ownership
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ownership_builder_dataset_id_requires_owner_and_tenant() {
    let f = fixture().await;
    let (o, t1, t2, d) = (
        f.owner.to_string(),
        f.t1.to_string(),
        f.t2.to_string(),
        f.o_t1.0.to_string(),
    );

    // Not the owner: refused before anything is deleted.
    let g = f.grantee.to_string();
    assert!(matches!(
        f.build(
            &["--user-id", &g, "--tenant-id", &t1, "--dataset-id", &d],
            RequestAuthorization::Ownership
        )
        .await,
        Err(CliError::Validation(_))
    ));
    // The owner, but in another tenant.
    assert!(matches!(
        f.build(
            &["--user-id", &o, "--tenant-id", &t2, "--dataset-id", &d],
            RequestAuthorization::Ownership
        )
        .await,
        Err(CliError::Validation(_))
    ));
    assert!(f.dataset_exists(f.o_t1.0).await);

    // The owner in its tenant.
    let (request, ..) = f
        .build(
            &["--user-id", &o, "--tenant-id", &t1, "--dataset-id", &d],
            RequestAuthorization::Ownership,
        )
        .await
        .unwrap();
    assert!(
        matches!(request.scope, DeleteScope::DatasetById { dataset_id } if dataset_id == f.o_t1.0)
    );
    f.plain().execute(&request).await.unwrap();
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await);
}

#[tokio::test]
async fn ownership_builder_dataset_name_resolves_to_id_in_tenant() {
    let f = fixture().await;
    let t1 = f.t1.to_string();
    let (request, ..) = f
        .build(
            &["--dataset-name", "shared", "--tenant-id", &t1],
            RequestAuthorization::Ownership,
        )
        .await
        .unwrap();
    assert!(
        matches!(request.scope, DeleteScope::DatasetById { dataset_id } if dataset_id == f.o_t1.0)
    );

    let (request, ..) = f
        .build(
            &["--dataset-name", "shared"],
            RequestAuthorization::Ownership,
        )
        .await
        .unwrap();
    assert!(
        matches!(request.scope, DeleteScope::DatasetById { dataset_id } if dataset_id == f.o_none.0),
        "no --tenant-id resolves the tenant-less row only"
    );

    // An item in a named dataset becomes `DataInDataset` by id.
    let x = f.o_t1.1.to_string();
    let (request, ..) = f
        .build(
            &[
                "--data-id",
                &x,
                "--dataset-name",
                "shared",
                "--tenant-id",
                &t1,
            ],
            RequestAuthorization::Ownership,
        )
        .await
        .unwrap();
    assert!(matches!(
        request.scope,
        DeleteScope::DataInDataset { dataset_id, data_id, .. }
            if dataset_id == f.o_t1.0 && data_id == f.o_t1.1
    ));
}

#[tokio::test]
async fn ownership_builder_all_is_callers_datasets() {
    let f = fixture().await;
    let t1 = f.t1.to_string();
    let (request, ..) = f
        .build(
            &["--all", "--tenant-id", &t1],
            RequestAuthorization::Ownership,
        )
        .await
        .unwrap();
    assert!(matches!(
        request.scope,
        DeleteScope::User { owner_id, tenant_id } if owner_id == f.owner && tenant_id == Some(f.t1)
    ));
    let result = f.plain().execute(&request).await.unwrap();
    assert_eq!(result.deleted_datasets, 1);
    assert!(!f.dataset_exists(f.o_t1.0).await);
    assert!(f.dataset_exists(f.o_t2.0).await, "other tenant untouched");
    assert!(f.dataset_exists(f.o_none.0).await, "tenant-less untouched");
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
fn scope_selection_rules() {
    let d = Uuid::new_v4().to_string();
    let u = Uuid::new_v4().to_string();
    let x = Uuid::new_v4().to_string();
    let err = |argv: &[&str]| validate_scope_selection(&delete_args(argv)).unwrap_err();

    assert!(format!("{}", err(&[])).contains("Specify exactly one delete scope"));
    assert!(matches!(
        err(&["--all", "--data-id", &x]),
        CliError::Validation(_)
    ));
    assert!(matches!(
        err(&["--all", "--dataset-id", &d]),
        CliError::Validation(_)
    ));
    assert!(matches!(
        err(&["--dataset-name", "n", "--delete-dataset-if-empty"]),
        CliError::Validation(_)
    ));

    validate_scope_selection(&delete_args(&["--data-id", &x, "--dataset-name", "n"]))
        .expect("an item in a named dataset is one target");
    validate_scope_selection(&delete_args(&["--user-id", &u, "--dataset-id", &d]))
        .expect("--user-id with a target names the acting user");
    validate_scope_selection(&delete_args(&["--user-id", &u]))
        .expect("--user-id alone is the User scope");
    validate_scope_selection(&delete_args(&[
        "--data-id",
        &x,
        "--delete-dataset-if-empty",
    ]))
    .expect("the flag applies with --data-id");
}

#[tokio::test]
async fn ownership_builder_user_id_alone_is_user_scope() {
    let f = fixture().await;
    let u = f.owner.to_string();
    let (request, ..) = f
        .build(&["--user-id", &u], RequestAuthorization::Ownership)
        .await
        .unwrap();
    assert!(matches!(
        request.scope,
        DeleteScope::User { owner_id, tenant_id: None } if owner_id == f.owner
    ));
}
