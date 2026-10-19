use sea_orm::{ConnectionTrait, DbBackend, QueryResult, Statement, Value};
use sea_orm_migration::prelude::*;

/// Does `table` already have `column`?
///
/// `SchemaManager::has_column` is unusable here: it dispatches on
/// *sea-orm-migration*'s own `sqlx-*` features, which this crate does not
/// enable (it only turns on `sea-orm/sqlx-sqlite` / `sea-orm/sqlx-postgres`),
/// so every arm is compiled out and the call panics with "feature is off".
pub(super) async fn has_column(
    manager: &SchemaManager<'_>,
    table: &str,
    column: &str,
) -> Result<bool, DbErr> {
    let backend = manager.get_database_backend();
    let sql = match backend {
        DbBackend::Sqlite => {
            "SELECT COUNT(*) AS cnt FROM pragma_table_info(?) WHERE name = ?".to_string()
        }
        DbBackend::Postgres => "SELECT COUNT(*) AS cnt FROM information_schema.columns \
             WHERE table_schema = current_schema() AND table_name = $1 AND column_name = $2"
            .to_string(),
        other => {
            return Err(DbErr::Custom(format!(
                "column-existence probe does not support the {other:?} backend"
            )));
        }
    };
    let row = probe_row(manager, &sql, [table.into(), column.into()]).await?;
    Ok(row.try_get::<i64>("", "cnt")? > 0)
}

/// Add `column` to `table`, or check that an existing one matches it.
pub(super) async fn add_column_if_missing(
    manager: &SchemaManager<'_>,
    table: &str,
    mut column: ColumnDef,
) -> Result<(), DbErr> {
    if has_column(manager, table, &column.get_column_name()).await? {
        return ensure_column_matches(manager, table, &column).await;
    }
    let alter = Table::alter()
        .table(Alias::new(table))
        .add_column(&mut column)
        .to_owned();
    manager.alter_table(alter).await
}

/// Fail unless `table`'s existing column has `expected`'s nullability and, on
/// Postgres, its type. SQLite types are not compared: SQLite does not enforce
/// them, and other writers name them differently (`CHAR(32)` for `text`).
async fn ensure_column_matches(
    manager: &SchemaManager<'_>,
    table: &str,
    expected: &ColumnDef,
) -> Result<(), DbErr> {
    let column = expected.get_column_name();
    let not_null = declares_not_null(expected);
    if column_is_not_null(manager, table, &column).await? != not_null {
        let wanted = if not_null { "NOT NULL" } else { "nullable" };
        return Err(DbErr::Custom(format!(
            "column {table}.{column} exists but is not {wanted}"
        )));
    }
    if manager.get_database_backend() == DbBackend::Postgres {
        ensure_postgres_type(manager, table, expected).await?;
    }
    Ok(())
}

/// Does `column` declare `NOT NULL`, directly or as a primary key?
fn declares_not_null(column: &ColumnDef) -> bool {
    column
        .get_column_spec()
        .iter()
        .any(|spec| matches!(spec, ColumnSpec::NotNull | ColumnSpec::PrimaryKey))
}

/// Is `table`'s existing `column` declared `NOT NULL`?
async fn column_is_not_null(
    manager: &SchemaManager<'_>,
    table: &str,
    column: &str,
) -> Result<bool, DbErr> {
    let sql = match manager.get_database_backend() {
        DbBackend::Sqlite => {
            "SELECT \"notnull\" <> 0 AS not_null FROM pragma_table_info(?) WHERE name = ?"
        }
        DbBackend::Postgres => {
            "SELECT is_nullable = 'NO' AS not_null \
             FROM information_schema.columns \
             WHERE table_schema = current_schema() AND table_name = $1 AND column_name = $2"
        }
        other => {
            return Err(DbErr::Custom(format!(
                "nullability probe does not support the {other:?} backend"
            )));
        }
    };
    let row = probe_row(manager, sql, [table.into(), column.into()]).await?;
    row.try_get::<bool>("", "not_null")
}

/// Run a schema probe that returns exactly one row.
async fn probe_row(
    manager: &SchemaManager<'_>,
    sql: &str,
    values: impl IntoIterator<Item = Value>,
) -> Result<QueryResult, DbErr> {
    let statement = Statement::from_sql_and_values(manager.get_database_backend(), sql, values);
    manager
        .get_connection()
        .query_one(statement)
        .await?
        .ok_or_else(|| DbErr::Custom(format!("schema probe returned no row: {sql}")))
}

/// Fail unless `table`'s existing Postgres column has `expected`'s base type,
/// ignoring modifiers such as a `varchar` length.
async fn ensure_postgres_type(
    manager: &SchemaManager<'_>,
    table: &str,
    expected: &ColumnDef,
) -> Result<(), DbErr> {
    let column = expected.get_column_name();
    let expected_type = postgres_type(expected)?;
    let sql = "SELECT format_type(a.atttypid, a.atttypmod) AS actual, \
         a.atttypid = CAST(CAST($3 AS text) AS regtype) AS matches \
         FROM pg_attribute a \
         WHERE a.attrelid = to_regclass(quote_ident($1)) AND a.attname = $2 \
         AND NOT a.attisdropped";
    let values = [
        table.into(),
        column.as_str().into(),
        expected_type.as_str().into(),
    ];
    let row = probe_row(manager, sql, values).await?;
    if !row.try_get::<bool>("", "matches")? {
        let actual: String = row.try_get("", "actual")?;
        return Err(DbErr::Custom(format!(
            "column {table}.{column} exists as {actual}, not {expected_type}"
        )));
    }
    Ok(())
}

/// `column`'s type as Postgres DDL spells it, e.g. `timestamp with time zone`.
fn postgres_type(column: &ColumnDef) -> Result<String, DbErr> {
    let column_type = column.get_column_type().ok_or_else(|| {
        DbErr::Custom(format!(
            "column {} declares no type",
            column.get_column_name()
        ))
    })?;
    let mut sql = String::new();
    PostgresQueryBuilder.prepare_column_type(column_type, &mut sql);
    Ok(sql)
}

/// Drop `column` from `table` if it exists.
pub(super) async fn drop_column_if_present(
    manager: &SchemaManager<'_>,
    table: &str,
    column: &str,
) -> Result<(), DbErr> {
    if !has_column(manager, table, column).await? {
        return Ok(());
    }
    let alter = Table::alter()
        .table(Alias::new(table))
        .drop_column(Alias::new(column))
        .to_owned();
    manager.alter_table(alter).await
}

/// Rename `from` to `to` in `table` if `from` exists.
pub(super) async fn rename_column_if_present(
    manager: &SchemaManager<'_>,
    table: &str,
    from: &str,
    to: &str,
) -> Result<(), DbErr> {
    if !has_column(manager, table, from).await? {
        return Ok(());
    }
    let alter = Table::alter()
        .table(Alias::new(table))
        .rename_column(Alias::new(from), Alias::new(to))
        .to_owned();
    manager.alter_table(alter).await
}

#[cfg(test)]
#[allow(
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use sea_orm::{ConnectionTrait, Database, DatabaseConnection, Statement};
    use sea_orm_migration::prelude::*;

    use super::{
        add_column_if_missing, drop_column_if_present, has_column, rename_column_if_present,
    };

    /// An in-memory database with one table, `t (id text)`.
    async fn db_with_table() -> DatabaseConnection {
        let db = Database::connect("sqlite::memory:")
            .await
            .expect("in-memory sqlite always connects");
        let sql = "CREATE TABLE t (id TEXT)".to_string();
        db.execute(Statement::from_string(db.get_database_backend(), sql))
            .await
            .expect("creating the test table must succeed");
        db
    }

    #[tokio::test]
    async fn has_column_tells_present_from_missing() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);

        assert!(has_column(&manager, "t", "id").await.expect("probe runs"));
        assert!(!has_column(&manager, "t", "nope").await.expect("probe runs"));
    }

    #[tokio::test]
    async fn add_column_if_missing_adds_once() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);
        let column = || ColumnDef::new(Alias::new("extra")).text().null().to_owned();

        add_column_if_missing(&manager, "t", column())
            .await
            .expect("adds the column");
        add_column_if_missing(&manager, "t", column())
            .await
            .expect("skips an existing column");

        assert!(
            has_column(&manager, "t", "extra")
                .await
                .expect("probe runs")
        );
    }

    #[tokio::test]
    async fn add_column_if_missing_rejects_other_nullability() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);
        let column = ColumnDef::new(Alias::new("id"))
            .text()
            .not_null()
            .to_owned();

        let error = add_column_if_missing(&manager, "t", column)
            .await
            .expect_err("a nullable `id` does not match NOT NULL");
        assert!(error.to_string().contains("t.id"), "{error}");
    }

    #[tokio::test]
    async fn add_column_if_missing_ignores_sqlite_types() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);
        let column = ColumnDef::new(Alias::new("id")).integer().null().to_owned();

        add_column_if_missing(&manager, "t", column)
            .await
            .expect("SQLite column types are not compared");
    }

    #[tokio::test]
    async fn drop_column_if_present_drops_once() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);
        let column = ColumnDef::new(Alias::new("extra")).text().null().to_owned();
        add_column_if_missing(&manager, "t", column)
            .await
            .expect("adds the column");

        drop_column_if_present(&manager, "t", "extra")
            .await
            .expect("drops the column");
        drop_column_if_present(&manager, "t", "extra")
            .await
            .expect("skips a missing column");

        assert!(
            !has_column(&manager, "t", "extra")
                .await
                .expect("probe runs")
        );
    }

    #[tokio::test]
    async fn rename_column_if_present_renames_once() {
        let db = db_with_table().await;
        let manager = SchemaManager::new(&db);

        rename_column_if_present(&manager, "t", "id", "key")
            .await
            .expect("renames the column");
        rename_column_if_present(&manager, "t", "id", "key")
            .await
            .expect("skips a missing column");

        assert!(!has_column(&manager, "t", "id").await.expect("probe runs"));
        assert!(has_column(&manager, "t", "key").await.expect("probe runs"));
    }
}
