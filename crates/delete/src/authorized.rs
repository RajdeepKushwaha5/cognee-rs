//! ACL-enforcing wrapper around [`DeleteService`].
//!
//! [`AuthorizedDeleteService`] checks that the calling principal holds the
//! `"delete"` permission on each target dataset before delegating to the
//! inner [`DeleteService`]. The plain `DeleteService` remains available for
//! edge/embedded deployments that do not require ACL enforcement.
//!
//! Every check is **tenant-scoped**: a grant counts only on a dataset whose
//! `tenant_id` equals the caller's tenant (strict equality, `None` admits only
//! tenant-less datasets), exactly like Python's
//! `get_all_user_permission_datasets` filter
//! (`cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:49-53@37f788bd2`).
//! A grant only in another tenant is denied with the same
//! [`DeleteError::PermissionDenied`] as no grant at all.

use std::collections::HashSet;
use std::sync::Arc;

use cognee_database::ops::acl::authorized_datasets_in_tenant;
use cognee_database::{AclDb, DeleteDb};
use cognee_models::Dataset;
use uuid::Uuid;

use crate::{DeleteError, DeletePreview, DeleteRequest, DeleteResult, DeleteScope, DeleteService};

/// Authorization-enforcing delete service.
///
/// Wraps a [`DeleteService`] and an [`AclDb`] implementation. Before every
/// `execute()` or `preview()` call the wrapper verifies that `principal_id`,
/// acting in `tenant_id`, holds `"delete"` permission on all affected
/// datasets (see the module docs for the tenant rule).
pub struct AuthorizedDeleteService {
    inner: DeleteService,
    acl_db: Arc<dyn AclDb>,
    database: Arc<dyn DeleteDb>,
}

impl AuthorizedDeleteService {
    /// Create a new authorized delete service.
    ///
    /// `database` must be the same connection used to construct the inner
    /// `DeleteService`, so that dataset resolution is consistent.
    pub fn new(inner: DeleteService, acl_db: Arc<dyn AclDb>, database: Arc<dyn DeleteDb>) -> Self {
        Self {
            inner,
            acl_db,
            database,
        }
    }

    /// Preview the deletion, checking ACL first.
    ///
    /// [`DeleteScope::All`] is previewed over the **union** of the caller's
    /// `delete`-authorized datasets in `tenant_id`
    /// ([`DeleteService::preview_datasets_by_id`]), so it agrees with what
    /// [`Self::execute`] removes — data shared between two of those datasets
    /// counts once, not zero times. By-name scopes are resolved in the
    /// caller's tenant and previewed by id (see [`Self::execute`]).
    pub async fn preview(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeletePreview, DeleteError> {
        if matches!(request.scope, DeleteScope::All) {
            let ids: Vec<Uuid> = self
                .deletable_in_tenant(principal_id, tenant_id)
                .await?
                .iter()
                .map(|ds| ds.id)
                .collect();
            return self.inner.preview_datasets_by_id(&ids).await;
        }
        let authorized = self.authorize(request, principal_id, tenant_id).await?;
        self.inner.preview(&authorized).await
    }

    /// Execute the deletion, checking ACL first.
    ///
    /// By-name scopes (`Dataset`, `Data` with a dataset name) are resolved
    /// **once**, tenant-strictly — owner, name and `dataset.tenant_id ==
    /// tenant_id` (`None` ⇔ NULL), Python's `get_dataset_ids` rule
    /// (`cognee/modules/data/methods/get_dataset_ids.py:30-34@37f788bd2`) —
    /// and the inner [`DeleteService`] is then driven **by that id**
    /// (`DatasetById` / `DataInDataset`). The permission check and the delete
    /// can therefore never act on different rows, and a same-named dataset in
    /// another tenant is never touched.
    ///
    /// [`DeleteScope::All`] is **not** a database-wide wipe here: like Python's
    /// `datasets.delete_all`, which enumerates
    /// `get_authorized_existing_datasets([], "delete", user)` and empties each
    /// (`cognee/api/v1/datasets/datasets.py:316-323@37f788bd2`), it deletes —
    /// by id, one dataset after another — every dataset the caller holds
    /// `delete` on in `tenant_id`, and leaves everything else (other tenants,
    /// ungranted datasets) alone. It therefore cannot be denied by a dataset
    /// the caller cannot see. The system-wide artifacts of the plain
    /// [`DeleteService`]'s `All` (whole graph/vector wipe, all search history)
    /// are not touched. Like Python it stops at the first failing dataset;
    /// the ones before it stay deleted and are reported in
    /// [`DeleteError::Partial`] (a failure on the first dataset is returned
    /// as is).
    pub async fn execute(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeleteResult, DeleteError> {
        if matches!(request.scope, DeleteScope::All) {
            let mut total = DeleteResult::default();
            let mut done = 0usize;
            for ds in self.deletable_in_tenant(principal_id, tenant_id).await? {
                match self.inner.execute(&by_id(request, ds.id)).await {
                    Ok(r) => {
                        total += r;
                        done += 1;
                    }
                    Err(e) if done == 0 => return Err(e),
                    Err(e) => {
                        tracing::warn!(
                            principal = %principal_id,
                            failed_dataset = %ds.id,
                            completed_datasets = done,
                            "delete-all stopped at the first failing dataset"
                        );
                        return Err(DeleteError::Partial {
                            completed: Box::new(total),
                            failed_dataset_id: ds.id,
                            source: Box::new(e),
                        });
                    }
                }
            }
            return Ok(total);
        }
        let authorized = self.authorize(request, principal_id, tenant_id).await?;
        self.inner.execute(&authorized).await
    }

    /// Existing datasets the principal holds `delete` on in `tenant_id`
    /// (strict tenant equality): one ACL listing plus one batch row lookup
    /// via the shared [`authorized_datasets_in_tenant`].
    async fn deletable_in_tenant(
        &self,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<Vec<Dataset>, DeleteError> {
        authorized_datasets_in_tenant(
            self.acl_db.as_ref(),
            self.database.as_ref(),
            principal_id,
            "delete",
            tenant_id,
        )
        .await
        .map_err(|e| DeleteError::Runtime(format!("ACL query failed: {e}")))
    }

    /// Authorize `request` and return the request to hand to the inner
    /// service: by-name scopes rewritten to the by-id scope of the dataset
    /// the check was made on; every other scope unchanged.
    async fn authorize(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeleteRequest, DeleteError> {
        let scope = match &request.scope {
            DeleteScope::Data {
                owner_id,
                data_id,
                dataset_name: Some(ds_name),
                delete_dataset_if_empty,
            } => {
                let dataset_id = self
                    .resolve_dataset_id(*owner_id, ds_name, tenant_id)
                    .await?;
                self.require_delete_permission(principal_id, tenant_id, dataset_id)
                    .await?;
                DeleteScope::DataInDataset {
                    dataset_id,
                    data_id: *data_id,
                    delete_dataset_if_empty: *delete_dataset_if_empty,
                }
            }
            DeleteScope::Dataset {
                owner_id,
                dataset_name,
            } => {
                let dataset_id = self
                    .resolve_dataset_id(*owner_id, dataset_name, tenant_id)
                    .await?;
                self.require_delete_permission(principal_id, tenant_id, dataset_id)
                    .await?;
                DeleteScope::DatasetById { dataset_id }
            }
            _ => {
                self.check_authorization(request, principal_id, tenant_id)
                    .await?;
                request.scope.clone()
            }
        };
        Ok(DeleteRequest {
            scope,
            mode: request.mode.clone(),
            memory_only: request.memory_only,
        })
    }

    /// Verify that `principal_id` has "delete" permission on all datasets
    /// that would be affected by the request, in `tenant_id`.
    async fn check_authorization(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<(), DeleteError> {
        match &request.scope {
            DeleteScope::Data {
                owner_id,
                data_id,
                dataset_name,
                ..
            } => {
                if let Some(ds_name) = dataset_name {
                    let dataset_id = self
                        .resolve_dataset_id(*owner_id, ds_name, tenant_id)
                        .await?;
                    self.require_delete_permission(principal_id, tenant_id, dataset_id)
                        .await?;
                } else {
                    // When no dataset_name is given, the inner service resolves
                    // all datasets linked to this data item that belong to the
                    // owner. We must check that the principal has delete
                    // permission on each of those datasets.
                    let datasets = self
                        .database
                        .list_datasets_for_data(*data_id)
                        .await
                        .map_err(|e| {
                            DeleteError::Runtime(format!(
                                "Failed to list datasets for data {data_id}: {e}"
                            ))
                        })?;

                    for ds in &datasets {
                        if ds.owner_id == *owner_id {
                            self.require_delete_permission(principal_id, tenant_id, ds.id)
                                .await?;
                        }
                    }
                }
            }
            DeleteScope::Dataset {
                owner_id,
                dataset_name,
            } => {
                let dataset_id = self
                    .resolve_dataset_id(*owner_id, dataset_name, tenant_id)
                    .await?;
                self.require_delete_permission(principal_id, tenant_id, dataset_id)
                    .await?;
            }
            // By id: Python's `get_authorized_dataset(user, dataset_id,
            // "delete")` — a `delete` grant on that dataset, not ownership
            // (SDK-936).
            DeleteScope::DataInDataset { dataset_id, .. }
            | DeleteScope::DatasetById { dataset_id } => {
                self.require_delete_permission(principal_id, tenant_id, *dataset_id)
                    .await?;
            }
            DeleteScope::User { owner_id } => {
                // For user-scoped deletion, verify the principal has delete
                // on all datasets that would be affected. The inner service
                // will list all datasets by owner. We check that the
                // principal's authorized set covers them. Collected into a set:
                // with tenant/role inheritance the authorized list can be
                // tenant-wide, and the cover check below is per dataset.
                //
                // The set is tenant-filtered like Python's `delete_all`, which
                // enumerates `get_authorized_existing_datasets([], "delete",
                // user)` (`cognee/api/v1/datasets/datasets.py:316-323@37f788bd2`
                // → `get_all_user_permission_datasets.py:49-53`). An owner
                // dataset in another tenant is therefore not covered and the
                // whole request fails closed.
                let authorized: HashSet<Uuid> = self
                    .acl_db
                    .authorized_dataset_ids_in_tenant(principal_id, "delete", tenant_id)
                    .await
                    .map_err(|e| DeleteError::Runtime(format!("ACL query failed: {e}")))?
                    .into_iter()
                    .collect();

                let owner_datasets = self
                    .database
                    .list_datasets_by_owner(*owner_id)
                    .await
                    .map_err(|e| {
                        DeleteError::Runtime(format!("Failed to list owner datasets: {e}"))
                    })?;

                // The denial names no dataset: the uncovered one may belong to
                // another tenant (or another owner) the caller cannot see.
                if let Some(ds) = owner_datasets
                    .iter()
                    .find(|ds| !authorized.contains(&ds.id))
                {
                    tracing::warn!(
                        principal = %principal_id,
                        dataset = %ds.id,
                        "user-scope delete denied: dataset not covered by the caller's delete grants in its tenant"
                    );
                    return Err(DeleteError::PermissionDenied(
                        "delete permission denied for the requested scope".to_string(),
                    ));
                }
            }
            // `execute` / `preview` expand `All` into per-dataset `DatasetById`
            // requests before authorizing, so `All` never reaches here. Fail
            // closed rather than allow it, should a refactor route it here.
            DeleteScope::All => {
                return Err(DeleteError::PermissionDenied(
                    "delete permission denied for the requested scope".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// `request` with its scope replaced by `DatasetById { dataset_id }`.
fn by_id(request: &DeleteRequest, dataset_id: Uuid) -> DeleteRequest {
    DeleteRequest {
        scope: DeleteScope::DatasetById { dataset_id },
        mode: request.mode.clone(),
        memory_only: request.memory_only,
    }
}

impl AuthorizedDeleteService {
    /// Look up a dataset by name + owner **in `tenant_id`** (strict, `None`
    /// ⇔ NULL; Python `get_dataset_ids.py:30-34@37f788bd2`) and return its ID.
    /// A same-named dataset in another tenant does not resolve.
    async fn resolve_dataset_id(
        &self,
        owner_id: Uuid,
        dataset_name: &str,
        tenant_id: Option<Uuid>,
    ) -> Result<Uuid, DeleteError> {
        let dataset = self
            .database
            .get_dataset_by_name_in_tenant(dataset_name, owner_id, tenant_id)
            .await
            .map_err(|e| {
                DeleteError::Runtime(format!("Failed to resolve dataset '{dataset_name}': {e}"))
            })?
            .ok_or_else(|| {
                DeleteError::Validation(format!(
                    "Dataset '{dataset_name}' was not found for owner {owner_id}"
                ))
            })?;

        Ok(dataset.id)
    }

    /// Check that the principal has "delete" permission on the dataset in
    /// `tenant_id` — Python `get_authorized_dataset(user, id, "delete")` →
    /// `get_specific_user_permission_datasets`
    /// (`cognee/modules/data/methods/get_authorized_dataset.py:11-29`,
    /// `cognee/modules/users/permissions/methods/get_specific_user_permission_datasets.py:27-38`
    /// @37f788bd2). No grant and a grant only in another tenant answer the
    /// same `PermissionDenied`.
    async fn require_delete_permission(
        &self,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
        dataset_id: Uuid,
    ) -> Result<(), DeleteError> {
        let has_perm = self
            .acl_db
            .has_permission_in_tenant(principal_id, dataset_id, "delete", tenant_id)
            .await
            .map_err(|e| DeleteError::Runtime(format!("ACL check failed: {e}")))?;

        if !has_perm {
            return Err(DeleteError::PermissionDenied(format!(
                "Principal {principal_id} does not have 'delete' permission on dataset {dataset_id}"
            )));
        }

        Ok(())
    }
}
