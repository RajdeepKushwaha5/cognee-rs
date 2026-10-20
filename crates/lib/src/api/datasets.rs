//! High-level dataset management facade.
//!
//! [`DatasetManager`] composes the existing `IngestDb`, `DeleteDb`, and `AclDb`
//! traits into a unified API matching the Python SDK's `datasets` class.

use std::collections::HashMap;
use std::sync::Arc;

use cognee_database::ops::acl::{
    authorized_datasets_in_tenant, grant_all_permissions_on_dataset_via_trait,
};
use cognee_database::{AclDb, DeleteDb, IngestDb, PipelineRunStatus};
use cognee_delete::{
    DeleteError, DeleteMode, DeleteRequest, DeleteResult, DeleteScope, DeleteService,
};
use cognee_ingestion::generate_dataset_id;
use cognee_models::{Data, Dataset};
use uuid::Uuid;

use super::error::DatasetError;

// The canonical permission list lives in `cognee_database::ops::acl::PERMISSION_NAMES`,
// which `grant_all_permissions_on_dataset_via_trait` iterates. Tests below assert
// against that same list rather than a local copy that could drift from it.

/// Combined trait for dataset operations (defined in `cognee-database`, so
/// every [`RelationalDb`](cognee_database::RelationalDb) upcasts to it).
pub use cognee_database::DatasetDb;

/// High-level facade for dataset CRUD operations.
///
/// Wraps the low-level DB traits with optional ACL enforcement, matching
/// the Python SDK's `datasets` class.
///
/// The caller-facing methods take the caller's `tenant_id` next to
/// `owner_id`; it must be the caller's own (authenticated) tenant. Where it is
/// applied, it is applied with **strict equality** (`None` admits only
/// datasets that have no tenant), like Python's
/// `dataset.tenant_id == user.tenant_id`
/// (`cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:49-53@37f788bd2`):
/// - [`Self::list_datasets`] and [`Self::delete_all`] apply it on **both**
///   paths (ACL and ownership fallback);
/// - [`Self::empty_dataset`] and [`Self::delete_data`] apply it on both
///   paths: under an ACL through the `delete` check, without one through the
///   ownership fallback (the dataset must be the caller's and in its tenant);
///   either way the delete then runs **by id**, never re-resolved by name;
/// - [`Self::list_data`] and [`Self::has_data`] apply it on both paths:
///   under an ACL through the `read` check, without one through the
///   ownership fallback (the dataset must be the caller's and in its tenant,
///   else [`DatasetError::NotFound`] — unknown, foreign and other-tenant ids
///   alike, so there is no existence oracle).
///
/// In short, every per-id method applies the tenant on both paths.
///
/// Under an ACL, "no grant", "a grant only in another tenant" and "unknown
/// id" all answer [`DatasetError::PermissionDenied`] — within each method
/// Python also answers one error for all three
/// (`get_specific_user_permission_datasets.py:27-38@37f788bd2`). Python's
/// HTTP status differs per method: `list_data` / `has_data` / `empty_dataset`
/// let `PermissionDeniedError` (403,
/// `cognee/modules/users/exceptions/exceptions.py:42-51`) propagate, while
/// `delete_data` re-raises it as `UnauthorizedDataAccessError` (**401**,
/// `cognee/api/v1/datasets/datasets.py:234-240`,
/// `cognee/modules/data/exceptions/exceptions.py:19-26` @37f788bd2). The
/// mapping to a status is the HTTP layer's choice.
pub struct DatasetManager {
    db: Arc<dyn DatasetDb>,
    acl_db: Option<Arc<dyn AclDb>>,
}

impl DatasetManager {
    /// Create a new `DatasetManager` without ACL enforcement.
    pub fn new(db: Arc<dyn DatasetDb>) -> Self {
        Self { db, acl_db: None }
    }

    /// Enable ACL enforcement using the given ACL database.
    pub fn with_acl(mut self, acl_db: Arc<dyn AclDb>) -> Self {
        self.acl_db = Some(acl_db);
        self
    }

    // ------------------------------------------------------------------
    // Read operations
    // ------------------------------------------------------------------

    /// List all datasets accessible to the given owner in `tenant_id`.
    ///
    /// When ACL is configured, only datasets the owner has "read" permission
    /// on **in `tenant_id`** are returned — the grants are the whole answer,
    /// including when they are empty. Without ACL, every dataset owned by the
    /// user whose tenant equals `tenant_id` is listed.
    ///
    /// Both paths apply Python's strict tenant rule: `datasets.list_datasets`
    /// is `get_authorized_existing_datasets([], "read", user)`
    /// (`cognee/api/v1/datasets/datasets.py:128-132@37f788bd2`), which keeps
    /// only `dataset.tenant_id == user.tenant_id`
    /// (`get_all_user_permission_datasets.py:49-53@37f788bd2`); a tenant-less
    /// caller (`None`) sees only tenant-less datasets.
    pub async fn list_datasets(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<Vec<Dataset>, DatasetError> {
        if let Some(acl) = &self.acl_db {
            let db: &dyn DeleteDb = self.db.as_ref();
            Ok(
                authorized_datasets_in_tenant(acl.as_ref(), db, owner_id, "read", tenant_id)
                    .await?,
            )
        } else {
            // Ownership fallback, strict like Python (`get_dataset_ids.py:34`,
            // `get_all_user_permission_datasets.py:49-53` @37f788bd2).
            Ok(IngestDb::list_datasets_by_owner(self.db.as_ref(), owner_id)
                .await?
                .into_iter()
                .filter(|ds| ds.tenant_id == tenant_id)
                .collect())
        }
    }

    /// List all data items in a dataset, with permission check.
    ///
    /// Results are sorted by `data_size` descending (largest first), matching
    /// Python SDK behaviour. The read is authorized first (Python
    /// `datasets.py:155@37f788bd2`), see [`Self::authorize_read`].
    pub async fn list_data(
        &self,
        dataset_id: Uuid,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<Vec<Data>, DatasetError> {
        self.authorize_read(owner_id, tenant_id, dataset_id).await?;
        let mut items = self.db.get_dataset_data(dataset_id).await?;
        items.sort_by_key(|b| std::cmp::Reverse(b.data_size));
        Ok(items)
    }

    /// Check whether a dataset contains any data items.
    ///
    /// Authorizes the read first (Python `datasets.py:164@37f788bd2`, see
    /// [`Self::authorize_read`]), then uses an efficient COUNT query instead
    /// of loading all records.
    pub async fn has_data(
        &self,
        dataset_id: Uuid,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<bool, DatasetError> {
        self.authorize_read(owner_id, tenant_id, dataset_id).await?;
        let count = self.db.count_dataset_data(dataset_id).await?;
        Ok(count > 0)
    }

    /// Get the latest pipeline status for each dataset, across all tracked pipelines.
    ///
    /// Returns a nested map `{ dataset_id → { pipeline_name → status } }`.
    /// Datasets with no pipeline runs for a given pipeline are omitted from the
    /// inner map (equivalent to Python's "not started" behaviour).
    pub async fn get_status(
        &self,
        dataset_ids: &[Uuid],
    ) -> Result<HashMap<Uuid, HashMap<String, PipelineRunStatus>>, DatasetError> {
        const PIPELINES: &[&str] = &["add_pipeline", "cognify_pipeline"];
        let mut statuses: HashMap<Uuid, HashMap<String, PipelineRunStatus>> =
            HashMap::with_capacity(dataset_ids.len());
        for &id in dataset_ids {
            for pipeline_name in PIPELINES {
                if let Some(status) = self
                    .db
                    .get_latest_pipeline_status(pipeline_name, id)
                    .await?
                {
                    statuses
                        .entry(id)
                        .or_default()
                        .insert(pipeline_name.to_string(), status);
                }
            }
        }
        Ok(statuses)
    }

    /// Scan a filesystem directory for dataset-like sub-directories.
    ///
    /// Returns the names of immediate child directories. This is a sync
    /// utility matching the Python SDK's `discover_datasets` method.
    pub fn discover_datasets(
        directory_path: &std::path::Path,
    ) -> Result<Vec<String>, DatasetError> {
        let mut datasets = Vec::new();
        for entry in std::fs::read_dir(directory_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && let Some(name) = entry.file_name().to_str()
            {
                datasets.push(name.to_owned());
            }
        }
        Ok(datasets)
    }

    // ------------------------------------------------------------------
    // Write / delete operations
    // ------------------------------------------------------------------

    /// Delete all data in a dataset (and the dataset record itself).
    ///
    /// Delegates to `DeleteService` with `DeleteScope::DatasetById`.
    ///
    /// Under an ACL, `delete` in `tenant_id` is checked **before** the dataset
    /// is looked up, so an unknown id is `PermissionDenied`, not `NotFound`
    /// (Python `empty_dataset` authorizes first, `datasets.py:190-193@37f788bd2`).
    pub async fn empty_dataset(
        &self,
        dataset_id: Uuid,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        delete_service: &DeleteService,
    ) -> Result<DeleteResult, DatasetError> {
        // With an ACL the `delete` grant is the authorization, so the dataset
        // is deleted by id whoever owns it (Python `empty_dataset`,
        // `datasets.py:184-215@37f788bd2`; SDK-936). Without one, ownership in
        // the caller's tenant is the authorization. Either way the delete is
        // by id: re-resolving the name would not be tenant-scoped.
        self.authorize_delete(owner_id, tenant_id, dataset_id)
            .await?;
        let request = DeleteRequest {
            scope: DeleteScope::DatasetById { dataset_id },
            mode: DeleteMode::Hard,
            memory_only: false,
        };
        Ok(delete_service.execute(&request).await?)
    }

    /// Delete a specific data item from a dataset.
    ///
    /// Delegates to `DeleteService` with `DeleteScope::DataInDataset`.
    ///
    /// Under an ACL, `delete` in `tenant_id` is checked **before** the dataset
    /// is looked up (Python `delete_data`, `datasets.py:234-240@37f788bd2`,
    /// which answers that denial as `UnauthorizedDataAccessError`, HTTP 401).
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors Python `delete_data`; the caller tenant is per call like `owner_id`"
    )]
    pub async fn delete_data(
        &self,
        dataset_id: Uuid,
        data_id: Uuid,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        mode: DeleteMode,
        delete_dataset_if_empty: bool,
        delete_service: &DeleteService,
    ) -> Result<DeleteResult, DatasetError> {
        // As in `empty_dataset` (Python `delete_data`,
        // `datasets.py:218-313@37f788bd2`): authorize, then delete by id.
        self.authorize_delete(owner_id, tenant_id, dataset_id)
            .await?;
        let request = DeleteRequest {
            scope: DeleteScope::DataInDataset {
                dataset_id,
                data_id,
                delete_dataset_if_empty,
            },
            mode,
            memory_only: false,
        };
        Ok(delete_service.execute(&request).await?)
    }

    /// Delete every dataset the caller may delete in `tenant_id`.
    ///
    /// Python `datasets.delete_all` enumerates
    /// `get_authorized_existing_datasets([], "delete", user)` — the
    /// `delete`-authorized datasets in the caller's tenant — and empties each
    /// by id (`cognee/api/v1/datasets/datasets.py:316-323@37f788bd2`). Here:
    /// - **ACL wired**: the datasets `authorized_dataset_ids_in_tenant(owner_id,
    ///   "delete", tenant_id)` names (a `read`-only grant is never deleted);
    /// - **no ACL**: the owner's datasets whose tenant equals `tenant_id`
    ///   (strict, as in [`Self::list_datasets`]).
    ///
    /// Each listed dataset is deleted **by id** (`DeleteScope::DatasetById`):
    /// under an ACL the listed datasets may belong to other owners, whose
    /// names the caller's own by-name lookup would not resolve.
    pub async fn delete_all(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        delete_service: &DeleteService,
    ) -> Result<Vec<DeleteResult>, DatasetError> {
        let datasets = if let Some(acl) = &self.acl_db {
            let db: &dyn DeleteDb = self.db.as_ref();
            authorized_datasets_in_tenant(acl.as_ref(), db, owner_id, "delete", tenant_id).await?
        } else {
            self.list_datasets(owner_id, tenant_id).await?
        };
        // One dataset after another, stopping at the first failure like
        // Python's loop; the datasets already deleted are reported in
        // `DeleteError::Partial` (a failure on the first is returned as is).
        let mut results: Vec<DeleteResult> = Vec::with_capacity(datasets.len());
        for ds in datasets {
            let request = DeleteRequest {
                scope: DeleteScope::DatasetById { dataset_id: ds.id },
                mode: DeleteMode::Hard,
                memory_only: false,
            };
            match delete_service.execute(&request).await {
                Ok(r) => results.push(r),
                Err(e) if results.is_empty() => return Err(e.into()),
                Err(e) => {
                    let mut completed = DeleteResult::default();
                    for r in results {
                        completed += r;
                    }
                    return Err(DatasetError::Delete(DeleteError::Partial {
                        completed: Box::new(completed),
                        failed_dataset_id: ds.id,
                        source: Box::new(e),
                    }));
                }
            }
        }
        Ok(results)
    }

    // ------------------------------------------------------------------
    // Create operations
    // ------------------------------------------------------------------

    /// Create a dataset with a deterministic ID matching Python's formula:
    ///   `uuid5(NAMESPACE_OID, f"{name}{user_id}{tenant_id}")`.
    ///
    /// Idempotent: if a dataset with the same deterministic ID already exists,
    /// returns the existing row.
    pub async fn create_dataset(
        &self,
        name: &str,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<Dataset, DatasetError> {
        let id = generate_dataset_id(name, owner_id, tenant_id);
        // Try to get existing dataset first (idempotent create).
        if let Some(existing) = self.db.get_dataset(id).await? {
            return Ok(existing);
        }
        let dataset = Dataset::new(name.to_string(), owner_id, tenant_id, id);
        Ok(self.db.create_dataset(dataset).await?)
    }

    /// Create a dataset and grant all four ACL permissions (`read`, `write`,
    /// `delete`, `share`) to the owner.
    pub async fn create_authorized_dataset(
        &self,
        name: &str,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        parent_user_id: Option<Uuid>,
    ) -> Result<Dataset, DatasetError> {
        let ds = self.create_dataset(name, owner_id, tenant_id).await?;
        let acl = self.acl_db.as_ref().ok_or(DatasetError::AclNotConfigured)?;

        // `grant_all_permissions_on_dataset_via_trait` ensures the principal
        // row exists before granting: the `acls.principal_id` FK references
        // `principals.id`, so a bare id would otherwise fail a foreign-key
        // constraint. Python's `give_permission_on_dataset` takes an
        // already-persisted `User`; this facade may be called with a bare id.
        // Both the upsert and the grants are idempotent.
        //
        // The same helper backs `POST /v1/datasets`
        // (`cognee_http_server::routers::datasets::create_new_dataset`), which
        // cannot call this facade — `cognee-http-server` deliberately does not
        // depend on `cognee` — so the two create paths are kept from
        // drifting by sharing this one grant implementation.
        grant_all_permissions_on_dataset_via_trait(acl.as_ref(), owner_id, ds.id).await?;
        if let Some(parent) = parent_user_id
            && parent != owner_id
        {
            grant_all_permissions_on_dataset_via_trait(acl.as_ref(), parent, ds.id).await?;
        }
        Ok(ds)
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// `read` on `dataset_id` in `tenant_id` (no-op without an ACL). No
    /// grant, a grant only in another tenant and an unknown id all answer
    /// `PermissionDenied` (Python `get_authorized_dataset.py:25-29`,
    /// `get_specific_user_permission_datasets.py:27-38` @37f788bd2).
    async fn check_read_permission(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        dataset_id: Uuid,
    ) -> Result<(), DatasetError> {
        if let Some(acl) = &self.acl_db
            && !acl
                .has_permission_in_tenant(owner_id, dataset_id, "read", tenant_id)
                .await?
        {
            return Err(DatasetError::PermissionDenied);
        }
        Ok(())
    }

    /// `delete` on `dataset_id` in `tenant_id`; same rule as
    /// [`Self::check_read_permission`].
    async fn check_delete_permission(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        dataset_id: Uuid,
    ) -> Result<(), DatasetError> {
        if let Some(acl) = &self.acl_db
            && !acl
                .has_permission_in_tenant(owner_id, dataset_id, "delete", tenant_id)
                .await?
        {
            return Err(DatasetError::PermissionDenied);
        }
        Ok(())
    }

    /// Authorize a delete of `dataset_id` by the caller.
    ///
    /// - **ACL**: `delete` in `tenant_id` (checked before any lookup, so an
    ///   unknown id is `PermissionDenied`), then the row must exist.
    /// - **No ACL** (ownership fallback): the row must exist, be owned by
    ///   `owner_id` and be in `tenant_id` (strict, `None` ⇔ NULL — Python
    ///   `get_dataset_ids.py:34@37f788bd2`); otherwise the same
    ///   `DeleteError::Validation("… not found …")` a by-name miss gave before.
    async fn authorize_delete(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        dataset_id: Uuid,
    ) -> Result<(), DatasetError> {
        if self.acl_db.is_some() {
            self.check_delete_permission(owner_id, tenant_id, dataset_id)
                .await?;
            self.require_dataset(dataset_id).await?;
            return Ok(());
        }
        let dataset = self.require_dataset(dataset_id).await?;
        if dataset.owner_id != owner_id || dataset.tenant_id != tenant_id {
            return Err(DatasetError::Delete(DeleteError::Validation(format!(
                "Dataset {dataset_id} was not found for owner {owner_id}"
            ))));
        }
        Ok(())
    }

    /// Authorize a read of `dataset_id` by the caller. Python always
    /// authorizes `list_data` / `has_data` with `get_authorized_dataset(user,
    /// dataset_id)` (`read`) (`cognee/api/v1/datasets/datasets.py:155, 164`,
    /// `get_authorized_dataset.py:25-29`,
    /// `get_specific_user_permission_datasets.py:27-38` @37f788bd2).
    ///
    /// - **ACL**: `read` in `tenant_id` ([`DatasetError::PermissionDenied`]).
    /// - **No ACL** (ownership replaces grants, as in
    ///   [`Self::authorize_delete`]): the row must exist, be owned by
    ///   `owner_id` and be in `tenant_id` (strict, `None` ⇔ NULL —
    ///   `get_dataset_ids.py:34@37f788bd2`), otherwise
    ///   [`DatasetError::NotFound`] for unknown, foreign and other-tenant ids
    ///   alike. Python's ownership read `get_dataset(user_id, dataset_id)`
    ///   likewise finds nothing for a dataset the user does not own
    ///   (`cognee/modules/data/methods/get_dataset.py:14-15@37f788bd2`).
    async fn authorize_read(
        &self,
        owner_id: Uuid,
        tenant_id: Option<Uuid>,
        dataset_id: Uuid,
    ) -> Result<(), DatasetError> {
        if self.acl_db.is_some() {
            return self
                .check_read_permission(owner_id, tenant_id, dataset_id)
                .await;
        }
        let dataset = self.require_dataset(dataset_id).await?;
        if dataset.owner_id != owner_id || dataset.tenant_id != tenant_id {
            return Err(DatasetError::NotFound);
        }
        Ok(())
    }

    async fn require_dataset(&self, id: Uuid) -> Result<Dataset, DatasetError> {
        self.db.get_dataset(id).await?.ok_or(DatasetError::NotFound)
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
    use cognee_database::{connect, initialize};
    use cognee_models::{Data, Dataset};
    use uuid::Uuid;

    /// Create a fresh in-memory SQLite database with migrations applied.
    async fn fresh_db() -> Arc<dyn cognee_database::RelationalDb> {
        let db = connect("sqlite::memory:")
            .await
            .expect("in-memory SQLite always connects");
        initialize(&db)
            .await
            .expect("migrations succeed on empty DB");
        Arc::new(db)
    }

    fn make_dataset(owner_id: Uuid) -> Dataset {
        Dataset::new(
            format!("test-dataset-{}", Uuid::new_v4()),
            owner_id,
            None,
            Uuid::new_v4(),
        )
    }

    fn make_data(owner_id: Uuid) -> Data {
        let id = Uuid::new_v4();
        let loc = format!("file:///tmp/test/{id}.txt");
        Data::builder(
            id,
            "test-data.txt",
            loc.as_str(),
            loc.as_str(),
            "txt",
            "text/plain",
            format!("{:x}", Uuid::new_v4()),
            owner_id,
        )
        .build()
    }

    fn make_data_with_size(owner_id: Uuid, size: i64) -> Data {
        let id = Uuid::new_v4();
        let loc = format!("file:///tmp/test/{id}.txt");
        Data::builder(
            id,
            "file.txt",
            loc.as_str(),
            loc.as_str(),
            "txt",
            "text/plain",
            format!("{:x}", Uuid::new_v4()),
            owner_id,
        )
        .data_size(size)
        .build()
    }

    #[tokio::test]
    async fn test_list_datasets_no_acl() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);

        // Insert dataset directly via IngestDb
        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let result = mgr
            .list_datasets(owner_id, None)
            .await
            .expect("list_datasets");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, ds.id);
    }

    #[tokio::test]
    async fn test_list_datasets_different_owner() {
        let db = fresh_db().await;
        let owner_a = Uuid::new_v4();
        let owner_b = Uuid::new_v4();

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(make_dataset(owner_a))
            .await
            .expect("create_dataset");
        ingest
            .create_dataset(make_dataset(owner_b))
            .await
            .expect("create_dataset");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let result_a = mgr
            .list_datasets(owner_a, None)
            .await
            .expect("list_datasets");
        assert_eq!(result_a.len(), 1);
        let result_b = mgr
            .list_datasets(owner_b, None)
            .await
            .expect("list_datasets");
        assert_eq!(result_b.len(), 1);
    }

    #[tokio::test]
    async fn test_has_data_empty_dataset() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        assert!(!mgr.has_data(ds.id, owner_id, None).await.expect("has_data"));
    }

    #[tokio::test]
    async fn test_has_data_with_data() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);
        let data = make_data(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");
        ingest.create_data(data.clone()).await.expect("create_data");
        ingest
            .attach_data_to_dataset(ds.id, data.id)
            .await
            .expect("attach_data");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        assert!(mgr.has_data(ds.id, owner_id, None).await.expect("has_data"));
    }

    #[tokio::test]
    async fn test_has_data_permission_denied_with_acl() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let other_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        // Grant read permission to owner only (via ACL). The test path uses
        // `MockAclDb` because this workspace provides no
        // `AclDb for DatabaseConnection`.
        let acl: Arc<dyn AclDb> = Arc::new(cognee_test_utils::MockAclDb::new());
        acl.ensure_principal(owner_id, "user")
            .await
            .expect("ensure_principal");
        acl.grant_permission(owner_id, ds.id, "read")
            .await
            .expect("grant_permission");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl);

        // Owner can check — should succeed.
        assert!(
            mgr.has_data(ds.id, owner_id, None).await.is_ok(),
            "owner must be able to call has_data"
        );

        // Other user gets PermissionDenied.
        let err = mgr
            .has_data(ds.id, other_id, None)
            .await
            .expect_err("must fail for unauthorized user");
        assert!(
            matches!(err, DatasetError::PermissionDenied),
            "expected PermissionDenied, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_list_data() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);
        let data = make_data(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");
        ingest.create_data(data.clone()).await.expect("create_data");
        ingest
            .attach_data_to_dataset(ds.id, data.id)
            .await
            .expect("attach_data");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let items = mgr
            .list_data(ds.id, owner_id, None)
            .await
            .expect("list_data");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, data.id);
    }

    #[tokio::test]
    async fn test_list_data_sorted_by_size_descending() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        // Create three data items with distinct sizes.
        let small = make_data_with_size(owner_id, 10);
        let large = make_data_with_size(owner_id, 1000);
        let medium = make_data_with_size(owner_id, 500);

        for d in [&small, &large, &medium] {
            ingest.create_data(d.clone()).await.expect("create_data");
            ingest
                .attach_data_to_dataset(ds.id, d.id)
                .await
                .expect("attach_data");
        }

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let items = mgr
            .list_data(ds.id, owner_id, None)
            .await
            .expect("list_data");
        assert_eq!(items.len(), 3);
        // Must be sorted largest first.
        assert_eq!(items[0].id, large.id, "largest must come first");
        assert_eq!(items[1].id, medium.id, "medium second");
        assert_eq!(items[2].id, small.id, "smallest last");
    }

    #[tokio::test]
    async fn test_get_status_no_runs() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let ds = make_dataset(owner_id);

        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let statuses = mgr.get_status(&[ds.id]).await.expect("get_status");
        // No pipeline runs recorded → the outer map should be empty.
        assert!(statuses.is_empty());
    }

    #[tokio::test]
    async fn test_discover_datasets() {
        let tmpdir = tempfile::tempdir().expect("create temp dir");
        std::fs::create_dir(tmpdir.path().join("dataset-a")).expect("create dir");
        std::fs::create_dir(tmpdir.path().join("dataset-b")).expect("create dir");
        // Create a file to verify it's excluded
        std::fs::write(tmpdir.path().join("not-a-dataset.txt"), "hello").expect("create file");

        let mut result =
            DatasetManager::discover_datasets(tmpdir.path()).expect("discover_datasets");
        result.sort();
        assert_eq!(result, vec!["dataset-a", "dataset-b"]);
    }

    #[tokio::test]
    async fn test_require_dataset_not_found() {
        let db = fresh_db().await;
        let mgr = DatasetManager::new(db as Arc<dyn DatasetDb>);
        let err = mgr.require_dataset(Uuid::new_v4()).await;
        assert!(matches!(err, Err(DatasetError::NotFound)));
    }

    #[tokio::test]
    async fn test_create_dataset_deterministic_id() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let tenant_id = Some(Uuid::new_v4());
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);

        let ds = mgr
            .create_dataset("my-ds", owner_id, tenant_id)
            .await
            .expect("create_dataset");

        let expected_id = generate_dataset_id("my-ds", owner_id, tenant_id);
        assert_eq!(ds.id, expected_id, "ID must match generate_dataset_id");
        assert_eq!(ds.name, "my-ds");
        assert_eq!(ds.owner_id, owner_id);
    }

    #[tokio::test]
    async fn test_create_dataset_idempotent() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);

        let ds1 = mgr
            .create_dataset("dup-ds", owner_id, None)
            .await
            .expect("first create");
        let ds2 = mgr
            .create_dataset("dup-ds", owner_id, None)
            .await
            .expect("second create");

        assert_eq!(ds1.id, ds2.id, "Idempotent: same ID returned on duplicate");

        // Ensure only one row exists
        let list = IngestDb::list_datasets_by_owner(db.as_ref(), owner_id)
            .await
            .unwrap();
        assert_eq!(list.len(), 1, "Only one row should exist");
    }

    #[tokio::test]
    async fn test_create_authorized_dataset_without_acl_errors() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let mgr = DatasetManager::new(db as Arc<dyn DatasetDb>);

        let result = mgr
            .create_authorized_dataset("auth-ds", owner_id, None, None)
            .await;
        assert!(
            matches!(result, Err(DatasetError::AclNotConfigured)),
            "Should error when ACL not configured"
        );
    }

    #[tokio::test]
    async fn test_create_authorized_dataset_grants_four_permissions() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let parent_id = Uuid::new_v4();
        let acl: Arc<dyn AclDb> = Arc::new(cognee_test_utils::MockAclDb::new());
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl.clone());

        let ds = mgr
            .create_authorized_dataset("auth-ds", owner_id, None, Some(parent_id))
            .await
            .expect("create_authorized_dataset");

        // Owner and parent both receive all four permissions on the dataset.
        for perm in cognee_database::ops::acl::PERMISSION_NAMES {
            assert!(
                acl.has_permission(owner_id, ds.id, perm).await.unwrap(),
                "owner must have '{perm}'"
            );
            assert!(
                acl.has_permission(parent_id, ds.id, perm).await.unwrap(),
                "parent must have '{perm}'"
            );
        }
    }

    #[tokio::test]
    async fn test_create_authorized_dataset_parent_equals_owner_no_duplicate() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let acl: Arc<dyn AclDb> = Arc::new(cognee_test_utils::MockAclDb::new());
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl.clone());

        // parent_user_id == owner_id: must succeed (idempotent) and grant once.
        let ds = mgr
            .create_authorized_dataset("auth-ds-self", owner_id, None, Some(owner_id))
            .await
            .expect("create_authorized_dataset with self-parent should succeed");

        for perm in cognee_database::ops::acl::PERMISSION_NAMES {
            assert!(
                acl.has_permission(owner_id, ds.id, perm).await.unwrap(),
                "owner must have '{perm}'"
            );
        }
    }

    // ------------------------------------------------------------------
    // Role / tenant inheritance (issue #206)
    //
    // Python's `datasets.list_datasets` → `get_authorized_existing_datasets
    // ([], "read", user)` and `get_authorized_dataset(user, id, perm)` walk
    // tenant and role grants. These tests grant to a role/tenant id and call
    // as the member; with the plain `AclDb` variants they go red.
    // ------------------------------------------------------------------

    /// Scenario: one dataset granted `read` directly, one only via a role,
    /// one not at all. Expected: `list_datasets` returns exactly the first two
    /// (exercises `authorized_dataset_ids_in_tenant`).
    #[tokio::test]
    async fn acl_list_datasets_includes_role_granted_dataset() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let member = Uuid::new_v4();
        let role = Uuid::new_v4();
        let direct = make_dataset(owner_id);
        let via_role = make_dataset(owner_id);
        let ungranted = make_dataset(owner_id);
        let ingest: &dyn IngestDb = db.as_ref();
        for ds in [&direct, &via_role, &ungranted] {
            ingest
                .create_dataset(ds.clone())
                .await
                .expect("create_dataset");
        }

        let acl = Arc::new(cognee_test_utils::MockAclDb::new());
        // A role only confers grants within a tenant the user belongs to
        // (AclDb's contract, and Python nests the role walk in the tenant loop).
        acl.add_user_to_tenant(member, Uuid::new_v4());
        acl.add_user_to_role(member, role);
        acl.grant_permission(member, direct.id, "read")
            .await
            .expect("grant direct");
        acl.grant_permission(role, via_role.id, "read")
            .await
            .expect("grant role");

        let mgr =
            DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl as Arc<dyn AclDb>);
        let mut listed: Vec<Uuid> = mgr
            .list_datasets(member, None)
            .await
            .expect("list_datasets")
            .into_iter()
            .map(|d| d.id)
            .collect();
        listed.sort();
        let mut expected = vec![direct.id, via_role.id];
        expected.sort();
        assert_eq!(listed, expected, "direct + role grants, nothing else");
    }

    /// Scenario: `read` granted only to a tenant the caller belongs to.
    /// Expected: `has_data` / `list_data` succeed (exercises
    /// `check_read_permission` → `has_permission_in_tenant`).
    #[tokio::test]
    async fn acl_read_grant_via_tenant_admits_has_data_and_list_data() {
        let db = fresh_db().await;
        let owner_id = Uuid::new_v4();
        let member = Uuid::new_v4();
        let tenant = Uuid::new_v4();
        let ds = make_dataset(owner_id);
        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        let acl = Arc::new(cognee_test_utils::MockAclDb::new());
        acl.add_user_to_tenant(member, tenant);
        acl.grant_permission(tenant, ds.id, "read")
            .await
            .expect("grant tenant");

        let mgr =
            DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl as Arc<dyn AclDb>);
        assert!(
            mgr.has_data(ds.id, member, None).await.is_ok(),
            "tenant-held read must admit has_data"
        );
        assert!(
            mgr.list_data(ds.id, member, None).await.is_ok(),
            "tenant-held read must admit list_data"
        );
    }

    /// Scenario: the dataset owner holds `delete` only through a role (no
    /// direct grant — e.g. revoked), and a second user holds `read` through
    /// another role. Expected: the owner's `empty_dataset` passes the gate
    /// (exercises `check_delete_permission` → `has_permission_in_tenant`);
    /// the reader is denied.
    ///
    /// `empty_dataset` deletes by id (`DeleteScope::DatasetById`) once the
    /// gate passes, so only the gate decides here.
    #[tokio::test]
    async fn acl_delete_grant_via_role_admits_empty_dataset() {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let member = Uuid::new_v4();
        let reader = Uuid::new_v4();
        let deleter_role = Uuid::new_v4();
        let reader_role = Uuid::new_v4();
        let ds = make_dataset(member);
        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");

        let acl = Arc::new(cognee_test_utils::MockAclDb::new());
        // A role only confers grants within a tenant the user belongs to
        // (AclDb's contract, and Python nests the role walk in the tenant loop).
        acl.add_user_to_tenant(member, Uuid::new_v4());
        acl.add_user_to_tenant(reader, Uuid::new_v4());
        acl.add_user_to_role(member, deleter_role);
        acl.add_user_to_role(reader, reader_role);
        acl.grant_permission(deleter_role, ds.id, "delete")
            .await
            .expect("grant delete to role");
        acl.grant_permission(reader_role, ds.id, "read")
            .await
            .expect("grant read to role");

        let storage = Arc::new(cognee_storage::MockStorage::new());
        let delete_service = DeleteService::new(
            storage as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );
        let mgr =
            DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl as Arc<dyn AclDb>);

        let err = mgr
            .empty_dataset(ds.id, reader, None, &delete_service)
            .await
            .expect_err("a role-held read grant must not authorize delete");
        assert!(
            matches!(err, DatasetError::PermissionDenied),
            "expected PermissionDenied, got {err:?}"
        );

        mgr.empty_dataset(ds.id, member, None, &delete_service)
            .await
            .expect("a role-held delete grant must authorize empty_dataset");
    }

    /// SDK-936. Scenario: Alice owns a dataset with one data item; Bob holds
    /// a direct `delete` grant on it, Carol holds only `read`.
    /// Expected (Python `datasets.delete_data` / `empty_dataset`, which
    /// authorize by `delete` grant on the dataset id, not ownership —
    /// `datasets.py:184-313` at f154b32): Carol is `PermissionDenied`; Bob
    /// can delete the item and then empty the dataset. Before the fix both
    /// of Bob's calls passed the gate and then failed resolving the dataset by
    /// `(name, Bob)` — the grant could never be used.
    #[tokio::test]
    async fn acl_delete_grant_lets_a_non_owner_delete() {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let alice = Uuid::new_v4();
        let bob = Uuid::new_v4();
        let carol = Uuid::new_v4();
        let ds = make_dataset(alice);
        let data = make_data(alice);
        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");
        ingest.create_data(data.clone()).await.expect("create_data");
        ingest
            .attach_data_to_dataset(ds.id, data.id)
            .await
            .expect("attach_data");
        let second = make_data(alice);
        ingest
            .create_data(second.clone())
            .await
            .expect("create_data");
        ingest
            .attach_data_to_dataset(ds.id, second.id)
            .await
            .expect("attach_data");

        let acl = Arc::new(cognee_test_utils::MockAclDb::new());
        acl.grant_permission(bob, ds.id, "delete")
            .await
            .expect("grant");
        acl.grant_permission(carol, ds.id, "read")
            .await
            .expect("grant");
        let delete_service = DeleteService::new(
            Arc::new(cognee_storage::MockStorage::new()) as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );
        let mgr =
            DatasetManager::new(db.clone() as Arc<dyn DatasetDb>).with_acl(acl as Arc<dyn AclDb>);

        let err = mgr
            .delete_data(
                ds.id,
                data.id,
                carol,
                None,
                DeleteMode::Soft,
                false,
                &delete_service,
            )
            .await
            .expect_err("read is not delete");
        assert!(matches!(err, DatasetError::PermissionDenied), "{err:?}");

        mgr.delete_data(
            ds.id,
            data.id,
            bob,
            None,
            DeleteMode::Soft,
            false,
            &delete_service,
        )
        .await
        .expect("a delete grant must let a non-owner delete a data item");
        let remaining: Vec<Uuid> = DeleteDb::get_dataset_data(db.as_ref(), ds.id)
            .await
            .expect("dataset data")
            .into_iter()
            .map(|d| d.id)
            .collect();
        assert_eq!(remaining, vec![second.id]);

        mgr.empty_dataset(ds.id, bob, None, &delete_service)
            .await
            .expect("a delete grant must let a non-owner empty the dataset");
        assert!(
            ingest.get_dataset(ds.id).await.expect("lookup").is_none(),
            "the dataset must be gone"
        );
    }

    /// SDK-936 guard: without an ACL nothing changes — ownership is the
    /// authorization, so a non-owner's delete resolves the dataset by
    /// `(name, caller)` and fails, and the owner's succeeds.
    #[tokio::test]
    async fn without_acl_delete_stays_owner_scoped() {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let alice = Uuid::new_v4();
        let bob = Uuid::new_v4();
        let ds = make_dataset(alice);
        let ingest: &dyn IngestDb = db.as_ref();
        ingest
            .create_dataset(ds.clone())
            .await
            .expect("create_dataset");
        let delete_service = DeleteService::new(
            Arc::new(cognee_storage::MockStorage::new()) as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);

        let err = mgr
            .empty_dataset(ds.id, bob, None, &delete_service)
            .await
            .expect_err("a non-owner must not delete without an ACL");
        assert!(
            matches!(
                err,
                DatasetError::Delete(cognee_delete::DeleteError::Validation(_))
            ),
            "{err:?}"
        );
        assert!(ingest.get_dataset(ds.id).await.expect("lookup").is_some());

        mgr.empty_dataset(ds.id, alice, None, &delete_service)
            .await
            .expect("the owner deletes");
        assert!(ingest.get_dataset(ds.id).await.expect("lookup").is_none());
    }

    // ------------------------------------------------------------------
    // Tenant scoping (W1/W2/W20). Python keeps only datasets with
    // `dataset.tenant_id == user.tenant_id`, strict, `None` ⇔ NULL
    // (`get_all_user_permission_datasets.py:49-53@37f788bd2`), and answers
    // one `PermissionDeniedError` for no grant / other tenant / unknown id
    // (`get_specific_user_permission_datasets.py:27-38@37f788bd2`).
    //
    // Fixtures set the tenant on the DB row (`create_dataset(.., Some(t))`)
    // **and** on the mock (`set_dataset_tenant`) so neither side passes
    // vacuously; each negative sits next to a same-tenant positive.
    // ------------------------------------------------------------------

    /// A manager over a fresh DB with a `MockAclDb`, plus a `DeleteService`.
    async fn tenant_fixture() -> (
        Arc<dyn cognee_database::RelationalDb>,
        Arc<cognee_test_utils::MockAclDb>,
        DatasetManager,
        DeleteService,
    ) {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let acl = Arc::new(cognee_test_utils::MockAclDb::new());
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>)
            .with_acl(acl.clone() as Arc<dyn AclDb>);
        let delete_service = DeleteService::new(
            Arc::new(cognee_storage::MockStorage::new()) as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );
        (db, acl, mgr, delete_service)
    }

    /// Create a dataset owned by `owner` in `tenant` (DB row + mock), with
    /// one attached data item. Returns `(dataset, data_id)`.
    async fn seed_tenant_dataset(
        db: &Arc<dyn cognee_database::RelationalDb>,
        acl: &cognee_test_utils::MockAclDb,
        owner: Uuid,
        tenant: Option<Uuid>,
    ) -> (Dataset, Uuid) {
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let ds = mgr
            .create_dataset(&format!("ds-{}", Uuid::new_v4()), owner, tenant)
            .await
            .expect("create_dataset");
        assert_eq!(ds.tenant_id, tenant, "the DB row carries the tenant");
        if let Some(t) = tenant {
            acl.set_dataset_tenant(ds.id, t);
        }
        let data = make_data(owner);
        let ingest: &dyn IngestDb = db.as_ref();
        ingest.create_data(data.clone()).await.expect("create_data");
        ingest
            .attach_data_to_dataset(ds.id, data.id)
            .await
            .expect("attach_data");
        (ds, data.id)
    }

    #[tokio::test]
    async fn acl_list_datasets_excludes_a_grant_in_another_tenant() {
        let (db, acl, mgr, _) = tenant_fixture().await;
        let user = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (in_t1, _) = seed_tenant_dataset(&db, &acl, user, Some(t1)).await;
        let (in_t2, _) = seed_tenant_dataset(&db, &acl, user, Some(t2)).await;
        let (untenanted, _) = seed_tenant_dataset(&db, &acl, user, None).await;
        for ds in [&in_t1, &in_t2, &untenanted] {
            acl.grant_permission(user, ds.id, "read").await.unwrap();
        }

        let ids = |v: Vec<Dataset>| v.into_iter().map(|d| d.id).collect::<Vec<_>>();
        assert_eq!(
            ids(mgr.list_datasets(user, Some(t1)).await.unwrap()),
            vec![in_t1.id]
        );
        assert_eq!(
            ids(mgr.list_datasets(user, None).await.unwrap()),
            vec![untenanted.id]
        );
    }

    #[tokio::test]
    async fn acl_read_checks_deny_a_grant_in_another_tenant() {
        let (db, acl, mgr, _) = tenant_fixture().await;
        let user = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (ds, _) = seed_tenant_dataset(&db, &acl, user, Some(t2)).await;
        acl.grant_permission(user, ds.id, "read").await.unwrap();

        for tenant in [Some(t1), None] {
            assert!(matches!(
                mgr.list_data(ds.id, user, tenant).await,
                Err(DatasetError::PermissionDenied)
            ));
            assert!(matches!(
                mgr.has_data(ds.id, user, tenant).await,
                Err(DatasetError::PermissionDenied)
            ));
        }
        assert_eq!(mgr.list_data(ds.id, user, Some(t2)).await.unwrap().len(), 1);
        assert!(mgr.has_data(ds.id, user, Some(t2)).await.unwrap());
    }

    #[tokio::test]
    async fn acl_delete_checks_deny_a_grant_in_another_tenant() {
        let (db, acl, mgr, delete_service) = tenant_fixture().await;
        let user = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (ds, data_id) = seed_tenant_dataset(&db, &acl, user, Some(t2)).await;
        acl.grant_permission(user, ds.id, "delete").await.unwrap();
        let ingest: &dyn IngestDb = db.as_ref();

        let err = mgr
            .delete_data(
                ds.id,
                data_id,
                user,
                Some(t1),
                DeleteMode::Soft,
                false,
                &delete_service,
            )
            .await
            .expect_err("a grant in T2 must not authorize a T1 delete_data");
        assert!(matches!(err, DatasetError::PermissionDenied), "{err:?}");
        let err = mgr
            .empty_dataset(ds.id, user, Some(t1), &delete_service)
            .await
            .expect_err("a grant in T2 must not authorize a T1 empty_dataset");
        assert!(matches!(err, DatasetError::PermissionDenied), "{err:?}");
        assert!(
            ingest.get_dataset(ds.id).await.unwrap().is_some(),
            "the dataset is intact after the denials"
        );
        mgr.delete_data(
            ds.id,
            data_id,
            user,
            Some(t2),
            DeleteMode::Soft,
            false,
            &delete_service,
        )
        .await
        .expect("delete_data in the dataset's tenant is admitted");
        mgr.empty_dataset(ds.id, user, Some(t2), &delete_service)
            .await
            .expect("empty_dataset in the dataset's tenant is admitted");
        assert!(ingest.get_dataset(ds.id).await.unwrap().is_none());
    }

    /// The error for "no grant" and for "a grant only in T2" is the same
    /// variant on every per-id method — the caller cannot tell them apart.
    #[tokio::test]
    async fn acl_cross_tenant_denial_is_the_same_as_no_grant() {
        let (db, acl, mgr, delete_service) = tenant_fixture().await;
        let user = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (granted, granted_data) = seed_tenant_dataset(&db, &acl, user, Some(t2)).await;
        let (ungranted, ungranted_data) = seed_tenant_dataset(&db, &acl, user, Some(t1)).await;
        for perm in ["read", "delete"] {
            acl.grant_permission(user, granted.id, perm).await.unwrap();
        }

        for (ds, data_id) in [(granted.id, granted_data), (ungranted.id, ungranted_data)] {
            let t = Some(t1);
            assert!(matches!(
                mgr.list_data(ds, user, t).await,
                Err(DatasetError::PermissionDenied)
            ));
            assert!(matches!(
                mgr.has_data(ds, user, t).await,
                Err(DatasetError::PermissionDenied)
            ));
            assert!(matches!(
                mgr.delete_data(
                    ds,
                    data_id,
                    user,
                    t,
                    DeleteMode::Soft,
                    false,
                    &delete_service
                )
                .await,
                Err(DatasetError::PermissionDenied)
            ));
            assert!(matches!(
                mgr.empty_dataset(ds, user, t, &delete_service).await,
                Err(DatasetError::PermissionDenied)
            ));
        }
    }

    /// Under an ACL the check runs before the lookup, so an unknown id is
    /// `PermissionDenied` like Python (`datasets.py:155, 190-193, 234-240`
    /// @37f788bd2), never `NotFound`.
    #[tokio::test]
    async fn acl_unknown_dataset_id_is_permission_denied_not_not_found() {
        let (_db, _acl, mgr, delete_service) = tenant_fixture().await;
        let user = Uuid::new_v4();
        let unknown = Uuid::new_v4();
        assert!(matches!(
            mgr.list_data(unknown, user, None).await,
            Err(DatasetError::PermissionDenied)
        ));
        assert!(matches!(
            mgr.empty_dataset(unknown, user, None, &delete_service)
                .await,
            Err(DatasetError::PermissionDenied)
        ));
        assert!(matches!(
            mgr.delete_data(
                unknown,
                Uuid::new_v4(),
                user,
                None,
                DeleteMode::Soft,
                false,
                &delete_service
            )
            .await,
            Err(DatasetError::PermissionDenied)
        ));
    }

    /// W20/D4: the ownership fallback is strict too — `None` sees only
    /// tenant-less datasets (`get_dataset_ids.py:34@37f788bd2`).
    #[tokio::test]
    async fn no_acl_list_datasets_is_tenant_strict() {
        let db = fresh_db().await;
        let acl = cognee_test_utils::MockAclDb::new();
        let owner = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (untenanted, _) = seed_tenant_dataset(&db, &acl, owner, None).await;
        let (in_t1, _) = seed_tenant_dataset(&db, &acl, owner, Some(t1)).await;
        let (_in_t2, _) = seed_tenant_dataset(&db, &acl, owner, Some(t2)).await;
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);

        let ids = |v: Vec<Dataset>| v.into_iter().map(|d| d.id).collect::<Vec<_>>();
        assert_eq!(
            ids(mgr.list_datasets(owner, None).await.unwrap()),
            vec![untenanted.id]
        );
        assert_eq!(
            ids(mgr.list_datasets(owner, Some(t1)).await.unwrap()),
            vec![in_t1.id]
        );
    }

    /// R1 regression. Scenario: no ACL; the owner has `"x"` in T1 (seeded
    /// **first**, so an unordered by-name lookup tends to return it) and a
    /// tenant-less `"x"`. A tenant-less `delete_all` lists only the
    /// tenant-less row. Expected: that row is deleted and the T1 row survives
    /// — the listed id is deleted, not re-resolved by name.
    #[tokio::test]
    async fn delete_all_deletes_the_listed_dataset_by_id_not_a_same_named_row() {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let owner = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let tenanted = mgr.create_dataset("x", owner, Some(t1)).await.unwrap();
        let untenanted = mgr.create_dataset("x", owner, None).await.unwrap();
        assert_ne!(tenanted.id, untenanted.id);
        let delete_service = DeleteService::new(
            Arc::new(cognee_storage::MockStorage::new()) as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );

        let results = mgr.delete_all(owner, None, &delete_service).await.unwrap();
        assert_eq!(results.len(), 1);
        let ingest: &dyn IngestDb = db.as_ref();
        assert!(
            ingest.get_dataset(untenanted.id).await.unwrap().is_none(),
            "the listed tenant-less dataset is deleted"
        );
        assert!(
            ingest.get_dataset(tenanted.id).await.unwrap().is_some(),
            "the T1 dataset with the same name must survive a tenant-less delete_all"
        );
    }

    /// Under an ACL, `delete_all` enumerates `delete`-authorized datasets in
    /// the caller's tenant (Python `datasets.py:316-323@37f788bd2`): a
    /// `read`-only grant and a `delete` grant in another tenant are left alone.
    #[tokio::test]
    async fn acl_delete_all_deletes_only_delete_authorized_datasets_in_the_callers_tenant() {
        let (db, acl, mgr, delete_service) = tenant_fixture().await;
        let caller = Uuid::new_v4();
        let other_owner = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let t2 = Uuid::new_v4();
        let (read_only, _) = seed_tenant_dataset(&db, &acl, other_owner, Some(t1)).await;
        let (deletable, _) = seed_tenant_dataset(&db, &acl, other_owner, Some(t1)).await;
        let (other_tenant, _) = seed_tenant_dataset(&db, &acl, other_owner, Some(t2)).await;
        acl.grant_permission(caller, read_only.id, "read")
            .await
            .unwrap();
        acl.grant_permission(caller, deletable.id, "delete")
            .await
            .unwrap();
        acl.grant_permission(caller, other_tenant.id, "delete")
            .await
            .unwrap();

        let results = mgr
            .delete_all(caller, Some(t1), &delete_service)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        let ingest: &dyn IngestDb = db.as_ref();
        assert!(ingest.get_dataset(deletable.id).await.unwrap().is_none());
        assert!(
            ingest.get_dataset(read_only.id).await.unwrap().is_some(),
            "a read-only grant must not be deleted"
        );
        assert!(
            ingest.get_dataset(other_tenant.id).await.unwrap().is_some(),
            "a delete grant in T2 must not be used from T1"
        );
    }

    /// Review loop 2. Without an ACL, `empty_dataset` / `delete_data` delete
    /// **by id** after an ownership + strict-tenant check. Fixture: `"x"` in
    /// T1 (seeded first) and a tenant-less `"x"`, same owner. The old by-name
    /// re-resolution could hit the T1 row.
    #[tokio::test]
    async fn no_acl_deletes_go_by_id_and_require_owner_and_tenant() {
        use cognee_database::DeleteDb;
        let db = fresh_db().await;
        let acl = cognee_test_utils::MockAclDb::new();
        let owner = Uuid::new_v4();
        let t1 = Uuid::new_v4();
        let mgr = DatasetManager::new(db.clone() as Arc<dyn DatasetDb>);
        let tenanted = mgr.create_dataset("x", owner, Some(t1)).await.unwrap();
        let untenanted = mgr.create_dataset("x", owner, None).await.unwrap();
        let (other, other_data) = seed_tenant_dataset(&db, &acl, owner, None).await;
        let delete_service = DeleteService::new(
            Arc::new(cognee_storage::MockStorage::new()) as Arc<dyn cognee_storage::StorageTrait>,
            db.clone() as Arc<dyn DeleteDb>,
        );
        let ingest: &dyn IngestDb = db.as_ref();

        // A T1 dataset is not the tenant-less caller's to delete.
        for err in [
            mgr.empty_dataset(tenanted.id, owner, None, &delete_service)
                .await
                .expect_err("tenant mismatch"),
            mgr.empty_dataset(untenanted.id, Uuid::new_v4(), None, &delete_service)
                .await
                .expect_err("owner mismatch"),
        ] {
            assert!(
                matches!(err, DatasetError::Delete(DeleteError::Validation(_))),
                "{err:?}"
            );
        }

        mgr.empty_dataset(untenanted.id, owner, None, &delete_service)
            .await
            .unwrap();
        assert!(ingest.get_dataset(untenanted.id).await.unwrap().is_none());
        assert!(
            ingest.get_dataset(tenanted.id).await.unwrap().is_some(),
            "the same-named T1 dataset survives"
        );

        mgr.delete_data(
            other.id,
            other_data,
            owner,
            None,
            DeleteMode::Soft,
            false,
            &delete_service,
        )
        .await
        .unwrap();
        assert!(
            DeleteDb::get_dataset_data(db.as_ref(), other.id)
                .await
                .unwrap()
                .is_empty()
        );
        mgr.empty_dataset(tenanted.id, owner, Some(t1), &delete_service)
            .await
            .expect("the T1 owner in T1 may delete it");
    }
}
