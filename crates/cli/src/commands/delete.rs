use std::io;
use std::sync::Arc;

use cognee::PipelineContext;
use cognee::api::DatasetRef;
use cognee::database::IngestDb;
use cognee::delete::{DeleteMode, DeleteRequest, DeleteScope};
use tracing::{info, warn};
use uuid::Uuid;

use crate::cli::{DeleteArgs, DeleteModeArg};
use crate::error::CliError;

pub fn run(args: DeleteArgs, cm: Arc<cognee::ComponentManager>) -> Result<(), CliError> {
    let dry_run = args.dry_run;
    let force = args.force;

    validate_scope_selection(&args)?;
    // OSS has no `AclDb` implementation: grant-based deletes
    // (`RequestAuthorization::Grant`) are served by the cloud CLI.
    if args.enforce_acl {
        return Err(CliError::Validation(
            "--enforce-acl is not supported by this build".to_string(),
        ));
    }

    let (owner_id, tenant_id) = {
        let settings = cm.settings();
        resolve_caller(&args, &settings.default_user_id)?
    };

    crate::teardown::run_command(Arc::clone(&cm), async {
        let database = cm
            .database()
            .await
            .map_err(|e| CliError::Runtime(format!("{e}")))?;

        let request = build_request_async(
            args,
            owner_id,
            tenant_id,
            database.clone() as Arc<dyn IngestDb>,
            RequestAuthorization::Ownership,
        )
        .await?;

        let service = super::build_delete_service(&cm).await?;

        let preview = service
            .preview(&request)
            .await
            .map_err(|error| CliError::Runtime(format!("Delete preview failed: {error}")))?;
        print_preview(&preview);

        if dry_run {
            return Ok(());
        }
        if !force {
            confirm_deletion()?;
        }

        let result = service
            .execute(&request)
            .await
            .map_err(|error| CliError::Runtime(format!("Delete execution failed: {error}")))?;
        print_result(&result);

        Ok(())
    })
}

pub fn print_preview(preview: &cognee::delete::DeletePreview) {
    info!(
        datasets = preview.datasets_to_delete,
        links = preview.dataset_links_to_delete,
        data = preview.data_to_delete,
        storage_files = preview.storage_files_to_delete,
        graph_nodes = preview.graph_nodes_to_delete,
        vector_points = preview.vector_points_to_delete,
        "delete preview"
    );
}

pub fn print_result(result: &cognee::delete::DeleteResult) {
    info!(
        deleted_datasets = result.deleted_datasets,
        deleted_links = result.deleted_dataset_links,
        deleted_data = result.deleted_data,
        deleted_storage = result.deleted_storage_files,
        deleted_graph_nodes = result.deleted_graph_nodes,
        deleted_vector_points = result.deleted_vector_points,
        deleted_orphan_entities = result.deleted_orphan_entities,
        deleted_orphan_entity_types = result.deleted_orphan_entity_types,
        deleted_orphan_edge_types = result.deleted_orphan_edge_types,
        warning_count = result.warnings.len(),
        "delete completed"
    );

    for warning in &result.warnings {
        warn!(warning = %warning, "delete warning");
    }
}

pub fn confirm_deletion() -> Result<(), CliError> {
    info!("This operation is irreversible. Continue? [y/N]: ");

    let mut confirmation = String::new();
    io::stdin()
        .read_line(&mut confirmation)
        .map_err(|error| CliError::Runtime(format!("Failed to read confirmation: {error}")))?;

    let answer = confirmation.trim().to_lowercase();
    if answer != "y" && answer != "yes" {
        info!("Deletion cancelled.");
        return Err(CliError::Validation(
            "Deletion cancelled by user".to_string(),
        ));
    }

    Ok(())
}

/// Check the delete flags. The **target selectors** are `--all`,
/// `--data-id` and `--dataset-name`/`--dataset-id`:
///
/// - `--all` cannot be combined with another target;
/// - `--data-id` may name its dataset (`--dataset-name`/`--dataset-id`): one
///   target, "this item in that dataset";
/// - with no target selector, `--user-id` is required and selects the whole
///   `User` scope; with a target selector it only names the **acting user**,
///   as Python's `--user-id` does (`cli/commands/delete_command.py:89-91@37f788bd2`);
/// - `--delete-dataset-if-empty` applies only with `--data-id`.
pub fn validate_scope_selection(args: &DeleteArgs) -> Result<(), CliError> {
    // --dataset-name and --dataset-id are mutually exclusive (enforced by
    // clap's `conflicts_with`), and together count as one dataset reference.
    let has_dataset = args.dataset_name.is_some() || args.dataset_id.is_some();
    let has_target = args.all || args.data_id.is_some() || has_dataset;

    if !has_target && args.user_id.is_none() {
        return Err(CliError::Validation(
            "Specify exactly one delete scope: --data-id (optionally with --dataset-name/--dataset-id), --dataset-name/--dataset-id, --user-id, or --all"
                .to_string(),
        ));
    }

    if args.all && (args.data_id.is_some() || has_dataset) {
        return Err(CliError::Validation(
            "--all cannot be combined with --data-id, --dataset-name or --dataset-id".to_string(),
        ));
    }

    if args.delete_dataset_if_empty && args.data_id.is_none() {
        return Err(CliError::Validation(
            "--delete-dataset-if-empty only applies with --data-id".to_string(),
        ));
    }

    Ok(())
}

/// Parse the optional `--dataset-name` / `--dataset-id` pair into a
/// [`DatasetRef`]. Returns `None` if neither was provided.
fn parse_dataset_ref(args: &DeleteArgs) -> Result<Option<DatasetRef>, CliError> {
    if let Some(id) = &args.dataset_id {
        let parsed = Uuid::parse_str(id).map_err(|error| {
            CliError::Validation(format!("Invalid --dataset-id '{id}': {error}"))
        })?;
        Ok(Some(DatasetRef::Id(parsed)))
    } else if let Some(name) = &args.dataset_name {
        Ok(Some(DatasetRef::Name(name.clone())))
    } else {
        Ok(None)
    }
}

/// Who authorizes the request [`build_request_async`] produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestAuthorization {
    /// No ACL: ownership in `tenant_id` is the authorization (plain
    /// `DeleteService`). Every dataset reference is resolved here, to a
    /// dataset the caller owns in `tenant_id`, and the request names it by id.
    Ownership,
    /// An `AuthorizedDeleteService` checks a `delete` grant in `tenant_id`
    /// (`--enforce-acl`). Dataset ids are passed through unresolved — the
    /// grant check decides, so another user's dataset is reachable by id;
    /// names stay the caller's own and are resolved by the service.
    Grant,
}

/// The acting user and tenant: `--user-id` (else `default_user_id`) and
/// `--tenant-id` (Python resolves the acting user from `--user-id`,
/// `cli/commands/delete_command.py:89-91@37f788bd2`).
pub fn resolve_caller(
    args: &DeleteArgs,
    default_user_id: &str,
) -> Result<(Uuid, Option<Uuid>), CliError> {
    let owner_id = match &args.user_id {
        Some(user_id) => Uuid::parse_str(user_id).map_err(|error| {
            CliError::Validation(format!("Invalid --user-id '{user_id}': {error}"))
        })?,
        None => Uuid::parse_str(default_user_id).map_err(|error| {
            CliError::Validation(format!(
                "Invalid default_user_id '{default_user_id}': {error}"
            ))
        })?,
    };
    let tenant_id = super::parse_tenant_id(args.tenant_id.as_deref())?;
    Ok((owner_id, tenant_id))
}

/// The id of the caller's own dataset `dref` in `tenant_id` (strict). A
/// dataset that does not exist, is another user's or is in another tenant gets
/// one `Validation` answer (no existence oracle); a failed lookup is a
/// `Runtime` error.
async fn own_dataset_id(
    dref: &DatasetRef,
    owner_id: Uuid,
    tenant_id: Option<Uuid>,
    db: &dyn IngestDb,
) -> Result<Uuid, CliError> {
    let found = match dref {
        DatasetRef::Id(id) => db
            .get_dataset(*id)
            .await
            .map_err(|e| CliError::Runtime(format!("Failed to look up dataset {id}: {e}")))?
            .filter(|ds| ds.owner_id == owner_id && ds.tenant_id == tenant_id),
        DatasetRef::Name(name) => db
            .get_dataset_by_name_in_tenant(name, owner_id, tenant_id)
            .await
            .map_err(|e| CliError::Runtime(format!("Failed to look up dataset '{name}': {e}")))?,
    };
    found.map(|ds| ds.id).ok_or_else(|| {
        let what = match dref {
            DatasetRef::Id(id) => id.to_string(),
            DatasetRef::Name(name) => format!("'{name}'"),
        };
        CliError::Validation(format!(
            "Dataset {what} was not found for this user in this tenant"
        ))
    })
}

/// Build the [`DeleteRequest`] for `args`, acting as `owner_id` in
/// `tenant_id`.
///
/// | Args | `Ownership` | `Grant` |
/// |---|---|---|
/// | `--dataset-id D` | D must be the caller's, in the tenant → `DatasetById` | `DatasetById{D}` (no lookup) |
/// | `--dataset-name N` | strict lookup → `DatasetById` | `Dataset{owner, N, tenant}` |
/// | `--data-id X` + dataset | resolved as above → `DataInDataset` | `DataInDataset` (id) / `Data{.., Some(N), tenant}` (name) |
/// | `--data-id X` alone | `Data{owner, X, None, tenant}` | validation error |
/// | `--user-id U` alone | `User{U, tenant}` | `User{U, tenant}` |
/// | `--all` | `User{owner, tenant}` | `All` |
///
/// Names stay **owner-scoped** in both modes — Python's CLI resolves
/// `get_datasets_by_name(name, user.id)` and then deletes in the caller's
/// tenant (`cli/commands/delete_command.py:96-101`,
/// `api/v1/datasets/datasets.py:190`,
/// `get_all_user_permission_datasets.py:49-53` @37f788bd2); grant reach goes
/// only through ids (`get_unique_dataset_id.py:15-24@37f788bd2`). `--all` is
/// Python's `datasets.delete_all(user)` (`delete_command.py:93-94`,
/// `datasets.py:316-323` @37f788bd2): the caller's datasets in its tenant,
/// never a store-wide wipe.
pub async fn build_request_async(
    args: DeleteArgs,
    owner_id: Uuid,
    tenant_id: Option<Uuid>,
    db: Arc<dyn IngestDb>,
    authorization: RequestAuthorization,
) -> Result<DeleteRequest, CliError> {
    validate_scope_selection(&args)?;
    let mode = match args.mode {
        DeleteModeArg::Soft => DeleteMode::Soft,
        DeleteModeArg::Hard => DeleteMode::Hard,
    };
    let dataset_ref = parse_dataset_ref(&args)?;
    let data_id = args
        .data_id
        .as_deref()
        .map(|raw| {
            Uuid::parse_str(raw).map_err(|error| {
                CliError::Validation(format!("Invalid --data-id '{raw}': {error}"))
            })
        })
        .transpose()?;
    let delete_dataset_if_empty = args.delete_dataset_if_empty;

    let scope = match authorization {
        RequestAuthorization::Ownership => {
            let dataset_id = match &dataset_ref {
                Some(dref) => Some(own_dataset_id(dref, owner_id, tenant_id, db.as_ref()).await?),
                None => None,
            };
            match (data_id, dataset_id) {
                (Some(data_id), Some(dataset_id)) => DeleteScope::DataInDataset {
                    dataset_id,
                    data_id,
                    delete_dataset_if_empty,
                },
                (Some(data_id), None) => DeleteScope::Data {
                    owner_id,
                    data_id,
                    dataset_name: None,
                    delete_dataset_if_empty,
                    tenant_id,
                },
                (None, Some(dataset_id)) => DeleteScope::DatasetById { dataset_id },
                // `--all` or `--user-id` alone: the caller's datasets in its
                // tenant.
                (None, None) => DeleteScope::User {
                    owner_id,
                    tenant_id,
                },
            }
        }
        RequestAuthorization::Grant => match (data_id, dataset_ref) {
            (Some(data_id), Some(DatasetRef::Id(dataset_id))) => DeleteScope::DataInDataset {
                dataset_id,
                data_id,
                delete_dataset_if_empty,
            },
            (Some(data_id), Some(DatasetRef::Name(dataset_name))) => DeleteScope::Data {
                owner_id,
                data_id,
                dataset_name: Some(dataset_name),
                delete_dataset_if_empty,
                tenant_id,
            },
            (Some(_), None) => {
                return Err(CliError::Validation(
                    "--data-id under --enforce-acl requires --dataset-name or --dataset-id"
                        .to_string(),
                ));
            }
            (None, Some(DatasetRef::Id(dataset_id))) => DeleteScope::DatasetById { dataset_id },
            (None, Some(DatasetRef::Name(dataset_name))) => DeleteScope::Dataset {
                owner_id,
                dataset_name,
                tenant_id,
            },
            (None, None) if args.all => DeleteScope::All,
            (None, None) => DeleteScope::User {
                owner_id,
                tenant_id,
            },
        },
    };

    Ok(DeleteRequest {
        scope,
        mode,
        memory_only: false,
    })
}
