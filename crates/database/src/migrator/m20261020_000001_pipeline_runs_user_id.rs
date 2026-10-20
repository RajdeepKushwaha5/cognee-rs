//! Adds `pipeline_runs.user_id` — the user who triggered the run — plus its
//! index.
//!
//! Python added the same column in alembic revision `a7f3c9e1b5d2`
//! (`cognee/alembic/versions/a7f3c9e1b5d2_extend_pipeline_runs_operation_record.py:36, 54@37f788bd2`):
//! nullable, no backfill, index `ix_pipeline_runs_user_id`, both added only
//! when missing. Rust and Python share one SQLite file and either SDK may
//! migrate it first, so this migration guards the column and the index the
//! same way rather than issuing a plain `ALTER TABLE` (an existing column must
//! still be nullable — `add_column_if_missing` checks it).
//!
//! Nullable with no default and no backfill: a row written before this
//! migration has no recorded triggerer, and `user_id IN (…)` never matches
//! `NULL`, so such a row is visible to a user-scoped listing only through its
//! dataset (Python's activity feed behaves the same).
//!
//! `text`, like every id column in this schema: it holds `uuid_hex` (32 hex
//! chars, no hyphens), which is what Python's SQLAlchemy `UUID` renders as on
//! SQLite.
use sea_orm_migration::prelude::*;

use super::helpers::{add_column_if_missing, drop_column_if_present};

#[derive(DeriveMigrationName)]
pub struct Migration;

const TABLE: &str = "pipeline_runs";
const COLUMN: &str = "user_id";
/// SQLAlchemy's default name for a single-column index — Python's name.
const INDEX: &str = "ix_pipeline_runs_user_id";

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let user_id = ColumnDef::new(Alias::new(COLUMN)).text().null().to_owned();
        add_column_if_missing(manager, TABLE, user_id).await?;
        manager
            .create_index(
                Index::create()
                    .if_not_exists()
                    .name(INDEX)
                    .table(Alias::new(TABLE))
                    .col(Alias::new(COLUMN))
                    .to_owned(),
            )
            .await
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        // Index first, then column — the reverse of `up`.
        manager
            .drop_index(
                Index::drop()
                    .if_exists()
                    .name(INDEX)
                    .table(Alias::new(TABLE))
                    .to_owned(),
            )
            .await?;
        drop_column_if_present(manager, TABLE, COLUMN).await
    }
}
