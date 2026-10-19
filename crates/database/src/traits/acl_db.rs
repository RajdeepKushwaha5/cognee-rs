use async_trait::async_trait;
use uuid::Uuid;

use crate::types::DatabaseError;

/// Access control list database trait.
///
/// Provides methods to check, grant, and revoke permissions on datasets
/// for principals (users, roles, tenants). All implementations must be
/// thread-safe for async multi-threaded usage.
///
/// This crate provides no `impl AclDb for DatabaseConnection`. Callers wire
/// ACL through `MockAclDb` (tests) or through a downstream implementation
/// (production deployments that provide their own `AclDb`).
#[async_trait]
pub trait AclDb: Send + Sync {
    /// Check if a principal has a specific permission on a dataset.
    ///
    /// Returns `true` if a matching ACL row exists (direct principal match).
    ///
    /// # Not an authorization check
    /// Direct grants only (no tenant or role inheritance) and **no tenant
    /// filter**: a direct grant on another tenant's dataset answers `true`.
    /// Use [`Self::has_permission_in_tenant`] to authorize data access.
    async fn has_permission(
        &self,
        principal_id: Uuid,
        dataset_id: Uuid,
        permission_name: &str,
    ) -> Result<bool, DatabaseError>;

    /// Return all dataset IDs for which the principal has the given permission.
    async fn authorized_dataset_ids(
        &self,
        principal_id: Uuid,
        permission_name: &str,
    ) -> Result<Vec<Uuid>, DatabaseError>;

    /// Grant a permission on a dataset to a principal.
    ///
    /// Idempotent: no-op if the grant already exists.
    async fn grant_permission(
        &self,
        principal_id: Uuid,
        dataset_id: Uuid,
        permission_name: &str,
    ) -> Result<(), DatabaseError>;

    /// Revoke a permission on a dataset from a principal.
    async fn revoke_permission(
        &self,
        principal_id: Uuid,
        dataset_id: Uuid,
        permission_name: &str,
    ) -> Result<(), DatabaseError>;

    /// Ensure a principal row exists (upsert by ID).
    async fn ensure_principal(
        &self,
        principal_id: Uuid,
        principal_type: &str,
    ) -> Result<(), DatabaseError>;

    /// Return every dataset ID the user holds `permission_name` on via a
    /// direct, tenant or role grant, across **every** tenant the user belongs
    /// to. Deduplicated.
    ///
    /// This is a building block, **not an authorization answer**: it applies
    /// no `dataset.tenant_id` filter, so it can include another tenant's
    /// datasets. Implementations use it to build
    /// [`Self::authorized_dataset_ids_in_tenant`]; callers that authorize data
    /// access must use that method or [`Self::has_permission_in_tenant`].
    async fn authorized_dataset_ids_with_roles(
        &self,
        user_id: Uuid,
        permission_name: &str,
    ) -> Result<Vec<Uuid>, DatabaseError>;

    /// [`Self::authorized_dataset_ids_with_roles`] restricted to datasets
    /// whose `tenant_id` equals `tenant_id` — **strict equality**, so `None`
    /// admits only datasets that have no tenant. Deduplicated.
    ///
    /// This is the answer to "which datasets may this user read" in Python:
    /// `get_all_user_permission_datasets` unions the direct, tenant and role
    /// grants (Python also unions the grants of the user's *agents*,
    /// `get_all_user_permission_datasets.py:26-30@37f788bd2` — not modelled by
    /// this trait, a pre-existing divergence) and then keeps only
    /// `dataset.tenant_id == user.tenant_id`
    /// (`cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:49-53`
    /// @37f788bd2). `authorized_dataset_ids_with_roles` deliberately spans every
    /// tenant the user belongs to, so a caller that authorizes data access must
    /// use this variant (SDK-689).
    ///
    /// Required rather than defaulted: the tenant lives on the dataset row,
    /// which only the implementation can see, and a default that skipped the
    /// filter would silently reopen the cross-tenant read.
    async fn authorized_dataset_ids_in_tenant(
        &self,
        user_id: Uuid,
        permission_name: &str,
        tenant_id: Option<Uuid>,
    ) -> Result<Vec<Uuid>, DatabaseError>;

    /// May `user_id` exercise `permission_name` on `dataset_id` *as a member
    /// of `tenant_id`*? True iff `dataset_id` is in
    /// [`Self::authorized_dataset_ids_in_tenant`]`(user_id, permission_name,
    /// tenant_id)`: direct ∪ tenant ∪ role grants (Python also counts the
    /// user's agents' grants; see the note there), then **strict**
    /// `dataset.tenant_id == tenant_id` (`None` admits only datasets with no
    /// tenant). An unknown dataset id is `false`.
    ///
    /// `tenant_id` must be the caller's own **authenticated** current tenant
    /// (Python reads `user.tenant_id` off the resolved user), never a value
    /// taken from the request: the filter is only as trustworthy as the tenant
    /// it is given.
    ///
    /// Python: `check_permission_on_dataset` / `get_authorized_dataset` →
    /// `get_specific_user_permission_datasets(user, perm, [id])`, which
    /// intersects the requested id with `get_all_user_permission_datasets` and
    /// raises one `PermissionDeniedError` (403) when it is missing — no grant,
    /// a grant only in another tenant, and an unknown id all answer the same.
    /// (`cognee/modules/users/permissions/methods/check_permission_on_dataset.py:27`,
    /// `get_specific_user_permission_datasets.py:27-38`,
    /// `get_all_user_permission_datasets.py:49-53`,
    /// `cognee/modules/users/exceptions/exceptions.py:42-51` @37f788bd2)
    ///
    /// The default is correct for every implementation because it is built on
    /// the required, tenant-filtered listing — but it costs **one full grant
    /// listing per point check** (and a caller that checks N datasets pays N
    /// listings). It exists so a provider stays registrable without it, and so
    /// `MockAclDb` exercises the reference semantics.
    ///
    /// # Production implementations MUST override this
    /// with a point query that gives the same answer: the `dataset` row with
    /// `id = dataset_id` and the strict tenant predicate
    /// (`tenant_id = $t`, or `tenant_id IS NULL` for `None`), joined with a
    /// direct / tenant / role grant of `permission_name` for `user_id`.
    async fn has_permission_in_tenant(
        &self,
        user_id: Uuid,
        dataset_id: Uuid,
        permission_name: &str,
        tenant_id: Option<Uuid>,
    ) -> Result<bool, DatabaseError> {
        Ok(self
            .authorized_dataset_ids_in_tenant(user_id, permission_name, tenant_id)
            .await?
            .contains(&dataset_id))
    }
}
