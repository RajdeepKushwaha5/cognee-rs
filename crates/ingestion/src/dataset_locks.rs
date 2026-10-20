//! The per-dataset lock registry, re-exported from
//! [`cognee_core::dataset_locks`] (where it moved so `cognee-delete` can take
//! the same locks without depending on this crate), plus the by-name lookup a
//! create needs.
//!
//! See [`cognee_core::dataset_locks`] for what the locks protect, how they are
//! keyed and how re-entrancy works.

pub use cognee_core::dataset_locks::{DatasetLockGuard, DatasetLocks};
use uuid::Uuid;

use crate::generate_dataset_id;

/// Acquire the lock for the dataset `name` identifies under this owner and
/// tenant — the same `uuid5` identity a create would write
/// ([`generate_dataset_id`]).
///
/// Use this on the by-name paths, which are the ones that can decide a
/// dataset is missing and race to create it.
pub async fn lock_dataset_identity(
    locks: &DatasetLocks,
    name: &str,
    owner_id: Uuid,
    tenant_id: Option<Uuid>,
) -> DatasetLockGuard {
    locks
        .lock(generate_dataset_id(name, owner_id, tenant_id))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scenario: the same name, owner and tenant are locked through
    /// `lock_dataset_identity`, while the equivalent `uuid5` id is locked
    /// directly.
    /// Expected: they contend — the by-name lock must key on the identity a
    /// create would write, not on some other hash of the name.
    /// Verification: hold the id lock, then assert the by-name lock cannot be
    /// acquired without waiting.
    #[tokio::test]
    async fn lock_dataset_identity_keys_on_the_deterministic_id() {
        let locks = DatasetLocks::new();
        let owner = Uuid::new_v4();
        let id = generate_dataset_id("ds", owner, None);

        let _held = locks.lock(id).await;

        let contended = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            lock_dataset_identity(&locks, "ds", owner, None),
        )
        .await;
        assert!(
            contended.is_err(),
            "lock_dataset_identity must contend with the deterministic id it maps to"
        );
    }
}
