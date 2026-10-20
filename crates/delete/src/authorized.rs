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
//!
//! The owner-scoped scopes (`Data`, `Dataset`, `User`) carry the tenant they
//! resolve in; it must be the caller's tenant, otherwise the request is denied
//! (it could never be authorized). Grant reach goes only through the by-id
//! scopes (`DataInDataset`, `DatasetById`): a by-name scope must name the
//! **caller's** own dataset (`owner_id == principal`, else denied before any
//! lookup), as in Python (`cognee/modules/data/methods/get_unique_dataset_id.py:15-24@37f788bd2`).
//! A data item without its dataset (`Data { dataset_name: None }`) is refused:
//! Python has no dataset-less item delete
//! (`cognee/api/v1/datasets/datasets.py:218-224@37f788bd2`,
//! `cognee/api/v1/forget/forget.py:161-162@37f788bd2`).

use std::collections::HashSet;
use std::sync::Arc;

use cognee_database::ops::acl::authorized_datasets_in_tenant;
use cognee_database::{AclDb, DeleteDb};
use cognee_models::Dataset;
use uuid::Uuid;

use crate::{DeleteError, DeletePreview, DeleteRequest, DeleteResult, DeleteScope, DeleteService};

/// The generic denial: it names no dataset, since the one that failed may
/// belong to a tenant or owner the caller cannot see.
const SCOPE_DENIED: &str = "delete permission denied for the requested scope";

/// Authorization-enforcing delete service.
///
/// Wraps a [`DeleteService`] and an [`AclDb`] implementation. Before every
/// `execute()` or `preview()` call the wrapper verifies that `principal_id`,
/// acting in `tenant_id`, holds `"delete"` permission on all affected
/// datasets, and that an owner-scoped request targets `tenant_id` (see the
/// module docs for the tenant rule).
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
    /// counts once, not zero times. [`DeleteScope::User`] likewise over the
    /// owner's datasets in `tenant_id` once the cover check passes. By-name
    /// scopes are resolved in the caller's tenant and previewed by id (see
    /// [`Self::execute`]).
    pub async fn preview(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeletePreview, DeleteError> {
        match &request.scope {
            DeleteScope::All => {
                let ids: Vec<Uuid> = self
                    .deletable_in_tenant(principal_id, tenant_id)
                    .await?
                    .iter()
                    .map(|ds| ds.id)
                    .collect();
                return self.inner.preview_datasets_by_id(&ids).await;
            }
            DeleteScope::User {
                owner_id,
                tenant_id: scope_tenant,
            } => {
                let ids = self
                    .covered_user_datasets(principal_id, tenant_id, *owner_id, *scope_tenant)
                    .await?;
                return self.inner.preview_datasets_by_id(&ids).await;
            }
            _ => {}
        }
        let authorized = self.authorize(request, principal_id, tenant_id).await?;
        self.inner.preview(&authorized).await
    }

    /// Execute the deletion, checking ACL first.
    ///
    /// By-name scopes (`Dataset`, `Data` with a dataset name) must target the
    /// caller's tenant (`scope.tenant_id == tenant_id`, else
    /// [`DeleteError::PermissionDenied`]) and are resolved **once**,
    /// tenant-strictly — owner, name and `dataset.tenant_id == tenant_id`
    /// (`None` ⇔ NULL), Python's `get_dataset_ids` rule
    /// (`cognee/modules/data/methods/get_dataset_ids.py:30-34@37f788bd2`) —
    /// and the inner [`DeleteService`] is then driven **by that id**
    /// (`DatasetById` / `DataInDataset`). The permission check and the delete
    /// can therefore never act on different rows, and a same-named dataset in
    /// another tenant is never touched. `Data` without a dataset name is
    /// refused with [`DeleteError::Validation`] (see the module docs).
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
    ///
    /// [`DeleteScope::User`] must target the caller's tenant; the owner's
    /// datasets in it must all be covered by the caller's `delete` grants in
    /// that tenant (else the generic `PermissionDenied`), and are then deleted
    /// by id with the same stop-at-first-error rule. Search history is never
    /// deleted, as in Python's `delete_all` (the history table has no tenant,
    /// so clearing it would reach the owner's other tenants).
    pub async fn execute(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeleteResult, DeleteError> {
        match &request.scope {
            DeleteScope::All => {
                let ids: Vec<Uuid> = self
                    .deletable_in_tenant(principal_id, tenant_id)
                    .await?
                    .iter()
                    .map(|ds| ds.id)
                    .collect();
                return self
                    .inner
                    .execute_datasets_by_id(&ids, request.mode.clone(), request.memory_only)
                    .await;
            }
            DeleteScope::User {
                owner_id,
                tenant_id: scope_tenant,
            } => {
                let ids = self
                    .covered_user_datasets(principal_id, tenant_id, *owner_id, *scope_tenant)
                    .await?;
                return self
                    .inner
                    .execute_datasets_by_id(&ids, request.mode.clone(), request.memory_only)
                    .await;
            }
            _ => {}
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

    /// Deny an owner-scoped request whose `scope_tenant` is not the caller's
    /// `tenant_id`: grants count only in the caller's tenant
    /// (`get_all_user_permission_datasets.py:49-53@37f788bd2`), so such a
    /// request could never be authorized.
    fn require_caller_tenant(
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
        scope_tenant: Option<Uuid>,
    ) -> Result<(), DeleteError> {
        if scope_tenant != tenant_id {
            tracing::warn!(
                principal = %principal_id,
                caller_tenant = ?tenant_id,
                scope_tenant = ?scope_tenant,
                "delete denied: the scope's tenant is not the caller's tenant"
            );
            return Err(DeleteError::PermissionDenied(SCOPE_DENIED.to_string()));
        }
        Ok(())
    }

    /// Deny a by-name scope (`Dataset`, `Data { dataset_name: Some }`) that
    /// is not the caller's own name in the caller's tenant, **before** any
    /// lookup. Python resolves a name only among the caller's own datasets
    /// (`cognee/modules/data/methods/get_dataset_ids.py:29-34@37f788bd2`, with
    /// `user.id`); grant reach goes only through ids
    /// (`get_unique_dataset_id.py:15-24@37f788bd2`). Checking first also keeps
    /// another owner's dataset names from answering "not found" vs "denied".
    fn require_own_name_scope(
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
        owner_id: Uuid,
        scope_tenant: Option<Uuid>,
    ) -> Result<(), DeleteError> {
        if owner_id != principal_id {
            tracing::warn!(
                principal = %principal_id,
                scope_owner = %owner_id,
                "delete denied: a by-name scope must name the caller's own dataset"
            );
            return Err(DeleteError::PermissionDenied(SCOPE_DENIED.to_string()));
        }
        Self::require_caller_tenant(principal_id, tenant_id, scope_tenant)
    }

    /// The `User { owner_id, scope_tenant }` target set, checked: the scope
    /// tenant must be the caller's, and every dataset `owner_id` owns in it
    /// must be covered by the caller's `delete` grants in that tenant —
    /// Python's `delete_all` enumerates exactly the tenant-filtered authorized
    /// set (`cognee/api/v1/datasets/datasets.py:316-323@37f788bd2` →
    /// `get_all_user_permission_datasets.py:49-53`). Returns the ids, in
    /// listing order.
    async fn covered_user_datasets(
        &self,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
        owner_id: Uuid,
        scope_tenant: Option<Uuid>,
    ) -> Result<Vec<Uuid>, DeleteError> {
        Self::require_caller_tenant(principal_id, tenant_id, scope_tenant)?;
        let owner_ids = self
            .inner
            .owner_dataset_ids_in_tenant(owner_id, tenant_id)
            .await?;
        // A set: with tenant/role inheritance the authorized list can be
        // tenant-wide, and the cover check is per dataset.
        let authorized: HashSet<Uuid> = self
            .acl_db
            .authorized_dataset_ids_in_tenant(principal_id, "delete", tenant_id)
            .await
            .map_err(|e| DeleteError::Runtime(format!("ACL query failed: {e}")))?
            .into_iter()
            .collect();
        if let Some(uncovered) = owner_ids.iter().find(|id| !authorized.contains(id)) {
            tracing::warn!(
                principal = %principal_id,
                dataset = %uncovered,
                "user-scope delete denied: dataset not covered by the caller's delete grants in its tenant"
            );
            return Err(DeleteError::PermissionDenied(SCOPE_DENIED.to_string()));
        }
        Ok(owner_ids)
    }

    /// Authorize `request` and return the request to hand to the inner
    /// service: by-name scopes rewritten to the by-id scope of the dataset
    /// the check was made on; by-id scopes unchanged. `All` and `User` are
    /// expanded by [`Self::execute`] / [`Self::preview`] and never get here.
    async fn authorize(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<DeleteRequest, DeleteError> {
        let scope = match &request.scope {
            DeleteScope::Data {
                dataset_name: None, ..
            } => {
                return Err(DeleteError::Validation(
                    "deleting a data item under an ACL requires its dataset (name or id)"
                        .to_string(),
                ));
            }
            DeleteScope::Data {
                owner_id,
                data_id,
                dataset_name: Some(ds_name),
                delete_dataset_if_empty,
                tenant_id: scope_tenant,
            } => {
                Self::require_own_name_scope(principal_id, tenant_id, *owner_id, *scope_tenant)?;
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
                tenant_id: scope_tenant,
            } => {
                Self::require_own_name_scope(principal_id, tenant_id, *owner_id, *scope_tenant)?;
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

    /// Verify that `principal_id` has "delete" permission, in `tenant_id`, on
    /// the dataset a by-id scope names.
    async fn check_authorization(
        &self,
        request: &DeleteRequest,
        principal_id: Uuid,
        tenant_id: Option<Uuid>,
    ) -> Result<(), DeleteError> {
        match &request.scope {
            // By id: Python's `get_authorized_dataset(user, dataset_id,
            // "delete")` — a `delete` grant on that dataset, not ownership
            // (SDK-936).
            DeleteScope::DataInDataset { dataset_id, .. }
            | DeleteScope::DatasetById { dataset_id } => {
                self.require_delete_permission(principal_id, tenant_id, *dataset_id)
                    .await
            }
            // By-name scopes are rewritten by `authorize`; `All` and `User`
            // are expanded by `execute` / `preview`. None reaches here — fail
            // closed rather than allow it, should a refactor route it here.
            DeleteScope::Data { .. }
            | DeleteScope::Dataset { .. }
            | DeleteScope::User { .. }
            | DeleteScope::All => Err(DeleteError::PermissionDenied(SCOPE_DENIED.to_string())),
        }
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
