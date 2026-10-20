use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, EnumIter, DeriveActiveEnum, Serialize, Deserialize)]
#[sea_orm(rs_type = "String", db_type = "Text")]
pub enum PipelineRunStatus {
    #[sea_orm(string_value = "DATASET_PROCESSING_INITIATED")]
    Initiated,
    #[sea_orm(string_value = "DATASET_PROCESSING_STARTED")]
    Started,
    #[sea_orm(string_value = "DATASET_PROCESSING_COMPLETED")]
    Completed,
    #[sea_orm(string_value = "DATASET_PROCESSING_ERRORED")]
    Errored,
}

#[derive(Clone, Debug, PartialEq, Eq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "pipeline_runs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: String,
    pub created_at: DateTimeUtc,
    pub status: PipelineRunStatus,
    #[sea_orm(indexed)]
    pub pipeline_run_id: String,
    pub pipeline_name: String,
    #[sea_orm(indexed)]
    pub pipeline_id: String,
    #[sea_orm(indexed, nullable)]
    pub dataset_id: Option<String>,
    #[sea_orm(column_type = "Json", nullable)]
    pub run_info: Option<Json>,
    /// The user who triggered the run (uuid hex), `NULL` for rows written
    /// before the column existed and for system writes (Python
    /// `PipelineRun.user_id`, `models/PipelineRun.py:73@37f788bd2`).
    #[sea_orm(indexed, nullable)]
    pub user_id: Option<String>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::dataset::Entity",
        from = "Column::DatasetId",
        to = "super::dataset::Column::Id"
    )]
    Dataset,
}

impl ActiveModelBehavior for ActiveModel {}
