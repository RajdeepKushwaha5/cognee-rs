//! ACL trait-level helper operations.
//!
//! The direct-`DatabaseConnection` implementations that backed the
//! `AclDb` blanket impl are not part of this crate: the auth entities
//! they depended on (`acl`, `permission`, `principal`, `user_role`,
//! `user_tenant`) no longer exist on the OSS schema.
//!
//! What remains here is the trait-only helper used by the OSS ingestion
//! pipeline (which still wires an `&dyn AclDb`) and the canonical
//! `PERMISSION_NAMES` list both halves of the split agree on.

use tracing::instrument;
use uuid::Uuid;

use crate::types::DatabaseError;

/// All permission names defined in the system.
pub const PERMISSION_NAMES: &[&str] = &["read", "write", "delete", "share"];

/// Grant all four permissions (read, write, delete, share) to a principal
/// on a dataset via the [`AclDb`](crate::traits::AclDb) trait.
///
/// Used by the ingestion pipeline to bless the dataset owner on every
/// `add` of a freshly-created dataset. Works with any `&dyn AclDb`
/// implementation, so callers can pair it with `MockAclDb` (tests) or with
/// a downstream ACL implementation (production deployments that provide
/// their own `AclDb`).
#[instrument(
    name = "cognee.db.relational.acl.grant_all_permissions_on_dataset_via_trait",
    level = "info",
    skip_all,
    err
)]
pub async fn grant_all_permissions_on_dataset_via_trait(
    acl_db: &dyn crate::traits::AclDb,
    principal_id: Uuid,
    dataset_id: Uuid,
) -> Result<(), DatabaseError> {
    acl_db.ensure_principal(principal_id, "user").await?;

    for perm_name in PERMISSION_NAMES {
        acl_db
            .grant_permission(principal_id, dataset_id, perm_name)
            .await?;
    }

    Ok(())
}

/// The **existing** datasets `user_id` holds `permission_name` on in
/// `tenant_id` — Python's `get_all_user_permission_datasets(user, perm)`
/// (direct ∪ tenant ∪ role grants, then strict
/// `dataset.tenant_id == user.tenant_id`,
/// `cognee/modules/users/permissions/methods/get_all_user_permission_datasets.py:49-53@37f788bd2`).
///
/// One ACL listing ([`AclDb::authorized_dataset_ids_in_tenant`](crate::traits::AclDb::authorized_dataset_ids_in_tenant))
/// plus one batch row lookup ([`DeleteDb::get_datasets_by_ids`](crate::traits::DeleteDb::get_datasets_by_ids)),
/// in the ACL's order. Grants on ids with no dataset row are dropped, and the
/// rows are re-checked against `tenant_id` (strict) so the answer never
/// depends on the ACL backend alone for the tenant filter.
///
/// The single implementation behind `DatasetManager::list_datasets`,
/// `DatasetManager::delete_all` and `AuthorizedDeleteService`'s
/// `DeleteScope::All`.
#[instrument(
    name = "cognee.db.relational.acl.authorized_datasets_in_tenant",
    level = "info",
    skip(acl, db)
)]
pub async fn authorized_datasets_in_tenant(
    acl: &dyn crate::traits::AclDb,
    db: &dyn crate::traits::DeleteDb,
    user_id: Uuid,
    permission_name: &str,
    tenant_id: Option<Uuid>,
) -> Result<Vec<cognee_models::Dataset>, DatabaseError> {
    let ids = acl
        .authorized_dataset_ids_in_tenant(user_id, permission_name, tenant_id)
        .await?;
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    Ok(db
        .get_datasets_by_ids(&ids)
        .await?
        .into_iter()
        .filter(|ds| ds.tenant_id == tenant_id)
        .collect())
}
