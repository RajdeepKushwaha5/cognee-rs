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

    /// Check permission considering role and tenant inheritance.
    ///
    /// Resolution order (mirrors Python `get_all_user_permission_datasets`):
    /// 1. Direct user ACL
    /// 2. Tenant-level ACL for each tenant the user belongs to
    /// 3. Role-level ACL for each role the user holds in those tenants
    async fn has_permission_with_roles(
        &self,
        user_id: Uuid,
        dataset_id: Uuid,
        permission_name: &str,
    ) -> Result<bool, DatabaseError>;

    /// Return all dataset IDs the user can access via direct, tenant, or
    /// role grants. Deduplicates results.
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
    /// grants and then keeps only `dataset.tenant_id == user.tenant_id`
    /// (`cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:49-55`
    /// at f154b32). `authorized_dataset_ids_with_roles` deliberately spans every
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
}
