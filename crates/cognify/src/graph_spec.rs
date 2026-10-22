//! The JSON graph-model spec DSL.
//!
//! Port of Python's `cognee/modules/graph_models` (`spec.py` + `compiler.py`).
//! A [`GraphSchemaSpec`] is the hand-writable JSON shape for a custom graph
//! model — entities, typed fields, and relations with a cardinality — that the
//! cognee frontend's graph-model editor also produces.
//! [`graph_spec_to_json_schema`] compiles it into the Pydantic-compatible JSON
//! Schema that [`CognifyConfig::graph_schema`](crate::CognifyConfig::graph_schema)
//! takes, which [`DynamicGraphSchema::compile`] turns into a runtime model.
//!
//! Field names are snake_case with camelCase aliases, so one JSON document
//! works for both the frontend editor and Python. Unknown keys are rejected.
//! `inverse` and `options.auto_type_nodes` are accepted for frontend parity but,
//! as in Python, not compiled into the schema.

use std::collections::HashSet;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::dynamic_graph::{
    DATA_POINT_FIELDS, DynamicGraphError, DynamicGraphSchema, python_str_repr,
};

/// Maximum number of entities one spec may declare (Python `MAX_ENTITIES`).
pub const MAX_ENTITIES: usize = 50;

/// Maximum number of fields one entity may declare (Python
/// `MAX_FIELDS_PER_ENTITY`).
pub const MAX_FIELDS_PER_ENTITY: usize = 40;

/// The field every entity's instances carry their type marker in.
const TYPE_MARKER_FIELD: &str = "is_type";

/// An invalid spec, or a compiled schema the runtime model compiler rejects.
///
/// The rule violations display Python's `ValueError` text verbatim, so a
/// binding caller sees the same message from either SDK. Shape errors carry
/// serde's message rather than Pydantic's.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GraphSpecError {
    /// The document does not have the spec's shape: a missing or unknown key,
    /// a value of the wrong type, or an unknown field `kind`.
    #[error("invalid graph spec: {0}")]
    Malformed(String),
    #[error("A graph spec must declare at least one entity.")]
    NoEntities,
    #[error("A graph spec declares {0} entities; the maximum is {MAX_ENTITIES}.")]
    TooManyEntities(usize),
    #[error("Enum field {entity}.{field} must declare at least one enum value.")]
    EmptyEnum { entity: String, field: String },
    #[error("Entity names must be unique.")]
    DuplicateEntityName,
    #[error(
        "Entity name {} must be a plain identifier (letters, digits, underscores; \
         starting with a letter).",
        python_str_repr(.0)
    )]
    InvalidEntityName(String),
    #[error(
        "Entity name {} collides with the generated type marker of entity {}.",
        python_str_repr(.name),
        python_str_repr(.entity)
    )]
    TypeMarkerCollision { name: String, entity: String },
    #[error("Root entity {} is not declared.", python_str_repr(.0))]
    UnknownRoot(String),
    #[error(
        "Entity {} declares {count} fields; the maximum is {MAX_FIELDS_PER_ENTITY}.",
        python_str_repr(.entity)
    )]
    TooManyFields { entity: String, count: usize },
    #[error("Entity {} has duplicate field names.", python_str_repr(.0))]
    DuplicateFieldName(String),
    #[error("Field {entity}.{} must be a plain identifier.", python_str_repr(.field))]
    InvalidFieldName { entity: String, field: String },
    #[error(
        "Field {entity}.{} collides with a DataPoint infrastructure field.",
        python_str_repr(.field)
    )]
    ReservedFieldName { entity: String, field: String },
    #[error("Field {0}.name must be a string primitive (it is the node's primary identifier).")]
    InvalidNameField(String),
    #[error(
        "Relation {entity}.{field} targets undeclared entity {}.",
        python_str_repr(.target)
    )]
    UnknownRelationTarget {
        entity: String,
        field: String,
        target: String,
    },
    #[error(
        "{attribute}{} {} on entity {} must reference 'name' or a declared primitive/enum field.",
        if *.attribute == "primary_label_field" { "" } else { " entry" },
        python_str_repr(.field),
        python_str_repr(.entity)
    )]
    InvalidFieldReference {
        attribute: &'static str,
        entity: String,
        field: String,
    },
    /// The compiled schema is not one the runtime model compiler accepts.
    #[error(transparent)]
    Compile(#[from] DynamicGraphError),
}

/// A custom graph model in the JSON spec DSL (Python `GraphSchemaSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSchemaSpec {
    #[serde(default)]
    pub options: GraphSchemaOptions,
    /// Entity used as the top-level model; defaults to the first entity.
    #[serde(default)]
    pub root: Option<String>,
    pub entities: Vec<EntitySpec>,
}

/// Spec-wide options (Python `GraphSchemaOptions`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GraphSchemaOptions {
    /// Accepted for frontend parity; not compiled into the schema.
    #[serde(
        default = "default_true",
        alias = "autoTypeNodes",
        deserialize_with = "lax_bool"
    )]
    pub auto_type_nodes: bool,
}

impl Default for GraphSchemaOptions {
    fn default() -> Self {
        Self {
            auto_type_nodes: true,
        }
    }
}

/// One entity (node type) of a spec (Python `EntitySpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EntitySpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default, alias = "primaryLabelField")]
    pub primary_label_field: Option<String>,
    /// Fields embedded for vector search; `None` or empty compiles to
    /// `["name"]`.
    #[serde(default, alias = "indexFields")]
    pub index_fields: Option<Vec<String>>,
    /// Fields whose values determine node identity, so two extracted nodes
    /// with equal values merge into one graph node. `None` compiles to
    /// `["name"]`; `Some(vec![])` opts out of merging.
    #[serde(default, alias = "identityFields")]
    pub identity_fields: Option<Vec<String>>,
    #[serde(default)]
    pub fields: Vec<FieldSpec>,
}

/// One field of an entity, discriminated by `kind` (Python `FieldSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum FieldSpec {
    Primitive(PrimitiveFieldSpec),
    Enum(EnumFieldSpec),
    Relation(RelationFieldSpec),
}

/// The scalar type of a primitive field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimitiveType {
    #[default]
    String,
    Number,
    Boolean,
    /// An ISO date string.
    Date,
}

/// Whether a relation holds one target or a list of them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Cardinality {
    #[default]
    One,
    Many,
}

/// A scalar field (Python `PrimitiveFieldSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrimitiveFieldSpec {
    pub name: String,
    #[serde(default, alias = "primitiveType")]
    pub primitive_type: PrimitiveType,
    #[serde(default, deserialize_with = "lax_bool")]
    pub required: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// A string field restricted to fixed values (Python `EnumFieldSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnumFieldSpec {
    pub name: String,
    #[serde(alias = "enumValues")]
    pub enum_values: Vec<String>,
    #[serde(default, deserialize_with = "lax_bool")]
    pub required: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// A field pointing at another entity (Python `RelationFieldSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationFieldSpec {
    pub name: String,
    pub relation: RelationSpec,
    /// Accepted for parity; a relation is never compiled as required.
    #[serde(default, deserialize_with = "lax_bool")]
    pub required: bool,
    #[serde(default)]
    pub description: Option<String>,
}

/// The target of a relation field (Python `RelationSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationSpec {
    #[serde(alias = "targetEntityName")]
    pub target_entity_name: String,
    #[serde(default)]
    pub cardinality: Cardinality,
    #[serde(default)]
    pub inverse: Option<InverseSpec>,
}

/// Reserved: accepted for frontend parity but not compiled into the schema
/// (Python `InverseSpec`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InverseSpec {
    #[serde(default = "default_true", deserialize_with = "lax_bool")]
    pub enabled: bool,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub cardinality: Option<Cardinality>,
}

fn default_true() -> bool {
    true
}

fn malformed(error: serde_json::Error) -> GraphSpecError {
    GraphSpecError::Malformed(error.to_string())
}

/// A boolean the way Pydantic's lax mode reads one: a JSON boolean, `0`/`1`
/// (integer or float), or one of the strings Pydantic accepts, any case.
fn lax_bool<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    let value = Value::deserialize(deserializer)?;
    let parsed = match &value {
        Value::Bool(flag) => Some(*flag),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        Value::String(text) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    };
    parsed.ok_or_else(|| serde::de::Error::custom(format!("expected a boolean, got {value}")))
}

impl FieldSpec {
    /// The field's name.
    pub fn name(&self) -> &str {
        match self {
            Self::Primitive(field) => &field.name,
            Self::Enum(field) => &field.name,
            Self::Relation(field) => &field.name,
        }
    }

    fn description(&self) -> Option<&str> {
        match self {
            Self::Primitive(field) => field.description.as_deref(),
            Self::Enum(field) => field.description.as_deref(),
            Self::Relation(field) => field.description.as_deref(),
        }
        .filter(|description| !description.is_empty())
    }
}

impl GraphSchemaSpec {
    /// Parse and validate a spec from a JSON value.
    ///
    /// # Errors
    /// [`GraphSpecError::Malformed`] when the value does not have the spec's
    /// shape, or any validation error of [`GraphSchemaSpec::validate`].
    pub fn from_value(value: &Value) -> Result<Self, GraphSpecError> {
        let spec = Self::parse(value)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Parse and validate a spec from a JSON string.
    ///
    /// # Errors
    /// As [`GraphSchemaSpec::from_value`].
    pub fn from_json(json: &str) -> Result<Self, GraphSpecError> {
        let spec: Self = serde_json::from_str(json).map_err(malformed)?;
        spec.validate()?;
        Ok(spec)
    }

    /// Deserialize a spec object, or a string holding one, without validating
    /// it.
    fn parse(value: &Value) -> Result<Self, GraphSpecError> {
        match value {
            Value::String(text) => serde_json::from_str(text).map_err(malformed),
            other => Self::deserialize(other).map_err(malformed),
        }
    }

    /// The entity compiled as the top-level model: `root`, else the first.
    ///
    /// `None` only for a spec that fails [`GraphSchemaSpec::validate`].
    pub fn root_entity(&self) -> Option<&EntitySpec> {
        match &self.root {
            None => self.entities.first(),
            Some(root) => self.entities.iter().find(|entity| entity.name == *root),
        }
    }

    /// Check the rules Python's `GraphSchemaSpec` validators enforce, in the
    /// same order, returning the first violation.
    ///
    /// Names are the safety gate of the compiled schema, so entity and field
    /// names are restricted to ASCII identifiers. This is slightly stricter
    /// than Python, whose `$`-anchored regex also accepts a name with one
    /// trailing newline.
    ///
    /// # Errors
    /// The [`GraphSpecError`] naming the first rule the spec breaks.
    pub fn validate(&self) -> Result<(), GraphSpecError> {
        // Field-level constraints (Pydantic `Field(min_length, max_length)`).
        if self.entities.is_empty() {
            return Err(GraphSpecError::NoEntities);
        }
        if self.entities.len() > MAX_ENTITIES {
            return Err(GraphSpecError::TooManyEntities(self.entities.len()));
        }
        for entity in &self.entities {
            for field in &entity.fields {
                if let FieldSpec::Enum(field) = field
                    && field.enum_values.is_empty()
                {
                    return Err(GraphSpecError::EmptyEnum {
                        entity: entity.name.clone(),
                        field: field.name.clone(),
                    });
                }
            }
        }

        let entity_names: HashSet<&str> = self.entities.iter().map(|e| e.name.as_str()).collect();
        if entity_names.len() != self.entities.len() {
            return Err(GraphSpecError::DuplicateEntityName);
        }
        for entity in &self.entities {
            let name = entity.name.as_str();
            if !is_identifier(name) {
                return Err(GraphSpecError::InvalidEntityName(name.to_string()));
            }
            // The compiler emits a "{Name}Type" marker definition per entity.
            if let Some(base) = name.strip_suffix("Type")
                && entity_names.contains(base)
            {
                return Err(GraphSpecError::TypeMarkerCollision {
                    name: name.to_string(),
                    entity: base.to_string(),
                });
            }
        }
        if let Some(root) = &self.root
            && !entity_names.contains(root.as_str())
        {
            return Err(GraphSpecError::UnknownRoot(root.clone()));
        }
        for entity in &self.entities {
            validate_entity(entity, &entity_names)?;
        }
        Ok(())
    }
}

fn validate_entity(
    entity: &EntitySpec,
    entity_names: &HashSet<&str>,
) -> Result<(), GraphSpecError> {
    if entity.fields.len() > MAX_FIELDS_PER_ENTITY {
        return Err(GraphSpecError::TooManyFields {
            entity: entity.name.clone(),
            count: entity.fields.len(),
        });
    }
    let field_names: HashSet<&str> = entity.fields.iter().map(FieldSpec::name).collect();
    if field_names.len() != entity.fields.len() {
        return Err(GraphSpecError::DuplicateFieldName(entity.name.clone()));
    }

    let mut value_field_names: HashSet<&str> = HashSet::from(["name"]);
    for field in &entity.fields {
        let name = field.name();
        if !is_identifier(name) {
            return Err(GraphSpecError::InvalidFieldName {
                entity: entity.name.clone(),
                field: name.to_string(),
            });
        }
        if is_reserved_field_name(name) {
            return Err(GraphSpecError::ReservedFieldName {
                entity: entity.name.clone(),
                field: name.to_string(),
            });
        }
        if name == "name" {
            // `name` is always emitted by the compiler as a required string;
            // an explicit declaration is only allowed as that exact shape.
            let is_string_primitive = matches!(
                field,
                FieldSpec::Primitive(PrimitiveFieldSpec {
                    primitive_type: PrimitiveType::String,
                    ..
                })
            );
            if !is_string_primitive {
                return Err(GraphSpecError::InvalidNameField(entity.name.clone()));
            }
            continue;
        }
        match field {
            FieldSpec::Relation(relation) => {
                let target = &relation.relation.target_entity_name;
                if !entity_names.contains(target.as_str()) {
                    return Err(GraphSpecError::UnknownRelationTarget {
                        entity: entity.name.clone(),
                        field: name.to_string(),
                        target: target.clone(),
                    });
                }
            }
            FieldSpec::Primitive(_) | FieldSpec::Enum(_) => {
                value_field_names.insert(name);
            }
        }
    }

    for (attribute, referenced) in [
        ("index_fields", &entity.index_fields),
        ("identity_fields", &entity.identity_fields),
    ] {
        for field in referenced.iter().flatten() {
            if !value_field_names.contains(field.as_str()) {
                return Err(GraphSpecError::InvalidFieldReference {
                    attribute,
                    entity: entity.name.clone(),
                    field: field.clone(),
                });
            }
        }
    }
    if let Some(label) = entity.primary_label_field.as_deref()
        && !label.is_empty()
        && !value_field_names.contains(label)
    {
        return Err(GraphSpecError::InvalidFieldReference {
            attribute: "primary_label_field",
            entity: entity.name.clone(),
            field: label.to_string(),
        });
    }
    Ok(())
}

/// Python's `^[A-Za-z][A-Za-z0-9_]*$`.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Python's `RESERVED_FIELD_NAMES`: every field of Python's `DataPoint` plus
/// the generated type-marker field. That is the stored base's keys plus
/// `ontology_uri`, a Python `DataPoint` field the Rust base does not carry
/// (and which a hand-written `graph_schema` may therefore declare).
fn is_reserved_field_name(name: &str) -> bool {
    matches!(name, TYPE_MARKER_FIELD | "ontology_uri") || DATA_POINT_FIELDS.contains(&name)
}

/// Compile a spec into the graph-model JSON Schema that
/// [`CognifyConfig::graph_schema`](crate::CognifyConfig::graph_schema) takes
/// (Python `graph_spec_to_json_schema`).
///
/// The output matches Python's key for key and in the same key order: every
/// entity and its `{Name}Type` marker in `$defs`, with the root entity's
/// schema also spread at the top level. `metadata` defaults carry
/// `index_fields` (empty or unset → `["name"]`) and `identity_fields` (unset →
/// `["name"]`), so nodes extracted from different chunks with the same name
/// merge into one graph node.
///
/// # Errors
/// Any [`GraphSchemaSpec::validate`] error; the spec is validated first.
pub fn graph_spec_to_json_schema(spec: &GraphSchemaSpec) -> Result<Value, GraphSpecError> {
    spec.validate()?;
    Ok(compile_validated(spec))
}

/// [`graph_spec_to_json_schema`] for a spec that passed
/// [`GraphSchemaSpec::validate`].
fn compile_validated(spec: &GraphSchemaSpec) -> Value {
    let mut defs = Map::new();
    for entity in &spec.entities {
        defs.insert(format!("{}Type", entity.name), entity_type_def(entity));
        defs.insert(entity.name.clone(), Value::Object(entity_schema(entity)));
    }
    let mut schema = Map::new();
    schema.insert("$defs".to_string(), Value::Object(defs));
    // `validate` guarantees a root: a declared `root` or a first entity.
    if let Some(root) = spec.root_entity() {
        schema.extend(entity_schema(root));
    }
    Value::Object(schema)
}

/// Validate a spec, compile it, and compile the result into the runtime
/// model (Python `graph_model_from_spec`).
///
/// # Errors
/// Any [`GraphSpecError`]; [`GraphSpecError::Compile`] when the compiled
/// schema is not a model Python could build — a root entity whose name is not
/// a class name, such as `person`.
pub fn graph_model_from_spec(spec: &GraphSchemaSpec) -> Result<DynamicGraphSchema, GraphSpecError> {
    let schema = graph_spec_to_json_schema(spec)?;
    Ok(DynamicGraphSchema::compile(&schema)?)
}

/// Read a caller-supplied spec — a spec object, or a string holding one — and
/// compile it into a graph schema. What the bindings' `graphSpec` option and
/// the CLI's `--graph-spec` carry.
///
/// # Errors
/// [`GraphSpecError::Malformed`] for a string that is not JSON or a value
/// that is not a spec, and any validation error.
pub fn graph_schema_from_spec_value(value: &Value) -> Result<Value, GraphSpecError> {
    graph_spec_to_json_schema(&GraphSchemaSpec::parse(value)?)
}

fn index_fields(entity: &EntitySpec) -> Vec<String> {
    match &entity.index_fields {
        Some(fields) if !fields.is_empty() => fields.clone(),
        _ => vec!["name".to_string()],
    }
}

fn identity_fields(entity: &EntitySpec) -> Vec<String> {
    entity
        .identity_fields
        .clone()
        .unwrap_or_else(|| vec!["name".to_string()])
}

/// The `{Name}Type` marker every instance carries as `is_type`. Its `name`
/// defaults to the entity name and identity is by name, so all markers of one
/// entity type merge into a single graph node.
fn entity_type_def(entity: &EntitySpec) -> Value {
    json!({
        "properties": {
            "name": {"default": entity.name, "type": "string"},
            "metadata": {
                "additionalProperties": true,
                "default": {
                    "index_fields": index_fields(entity),
                    "identity_fields": ["name"],
                },
                "type": "object",
            },
        },
        "title": format!("{}Type", entity.name),
        "type": "object",
    })
}

fn entity_schema(entity: &EntitySpec) -> Map<String, Value> {
    let mut properties = Map::new();
    // `name` is the primary node identifier — always present.
    properties.insert("name".to_string(), json!({"type": "string"}));
    properties.insert(
        TYPE_MARKER_FIELD.to_string(),
        json!({"$ref": format!("#/$defs/{}Type", entity.name)}),
    );
    // `metadata` tells cognee which fields to index and merge on.
    properties.insert(
        "metadata".to_string(),
        json!({
            "additionalProperties": true,
            "default": {
                "index_fields": index_fields(entity),
                "identity_fields": identity_fields(entity),
            },
            "type": "object",
        }),
    );
    let mut required = vec![json!("name"), json!(TYPE_MARKER_FIELD)];

    for field in &entity.fields {
        if field.name() == "name" {
            continue;
        }
        let mut field_schema = match field {
            FieldSpec::Primitive(primitive) => {
                let kind = match primitive.primitive_type {
                    PrimitiveType::String | PrimitiveType::Date => "string",
                    PrimitiveType::Number => "number",
                    PrimitiveType::Boolean => "boolean",
                };
                json!({"type": kind})
            }
            FieldSpec::Enum(choices) => json!({"enum": choices.enum_values, "type": "string"}),
            FieldSpec::Relation(relation) => {
                let reference = format!("#/$defs/{}", relation.relation.target_entity_name);
                match relation.relation.cardinality {
                    Cardinality::Many => json!({
                        "default": [],
                        "items": {"$ref": reference},
                        "type": "array",
                    }),
                    Cardinality::One => json!({"$ref": reference}),
                }
            }
        };
        if let (Some(description), Value::Object(map)) = (field.description(), &mut field_schema) {
            map.insert("description".to_string(), json!(description));
        }
        properties.insert(field.name().to_string(), field_schema);
        let is_required = match field {
            FieldSpec::Primitive(field) => field.required,
            FieldSpec::Enum(field) => field.required,
            FieldSpec::Relation(_) => false,
        };
        if is_required {
            required.push(json!(field.name()));
        }
    }

    let mut schema = Map::new();
    schema.insert("properties".to_string(), Value::Object(properties));
    schema.insert("required".to_string(), Value::Array(required));
    schema.insert("title".to_string(), json!(entity.name));
    schema.insert("type".to_string(), json!("object"));
    if let Some(description) = entity.description.as_deref().filter(|d| !d.is_empty()) {
        schema.insert("description".to_string(), json!(description));
    }
    schema
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;

    /// One row of `tests/fixtures/graph_spec/graph_spec_table.json`.
    #[derive(Deserialize)]
    struct Row {
        name: String,
        spec: Value,
        schema: Option<Value>,
        buildable: Option<bool>,
        error: Option<String>,
    }

    fn table() -> Vec<Row> {
        serde_json::from_str(include_str!(
            "../tests/fixtures/graph_spec/graph_spec_table.json"
        ))
        .unwrap()
    }

    /// Every spec Python accepts compiles to Python's schema, key for key and
    /// in the same key order, and builds a runtime model exactly when Python's
    /// `graph_model_from_spec` builds one.
    #[test]
    fn accepted_specs_compile_to_python_schema() {
        let rows: Vec<Row> = table().into_iter().filter(|r| r.schema.is_some()).collect();
        assert!(rows.len() >= 5);
        for row in rows {
            let expected = row.schema.unwrap();
            let spec = GraphSchemaSpec::from_value(&row.spec).unwrap_or_else(|e| {
                panic!("{}: Rust rejected a spec Python accepts: {e}", row.name)
            });
            let schema = graph_spec_to_json_schema(&spec).unwrap();
            assert_eq!(schema, expected, "{}: compiled schema differs", row.name);
            assert_eq!(
                serde_json::to_string(&schema).unwrap(),
                serde_json::to_string(&expected).unwrap(),
                "{}: key order differs",
                row.name
            );
            assert_eq!(
                graph_schema_from_spec_value(&row.spec).unwrap(),
                expected,
                "{}: value entry point differs",
                row.name
            );
            assert_eq!(
                graph_model_from_spec(&spec).is_ok(),
                row.buildable.unwrap(),
                "{}: buildability differs from Python",
                row.name
            );
        }
    }

    /// Every spec Python rejects is rejected; a rule violation (Python's
    /// `Value error, ...`) carries Python's message verbatim.
    #[test]
    fn rejected_specs_fail_with_python_messages() {
        let rows: Vec<Row> = table().into_iter().filter(|r| r.error.is_some()).collect();
        assert!(rows.len() >= 40);
        for row in rows {
            let expected = row.error.unwrap();
            let error = GraphSchemaSpec::from_value(&row.spec)
                .err()
                .unwrap_or_else(|| panic!("{}: Rust accepted a spec Python rejects", row.name));
            if let Some(message) = expected.strip_prefix("Value error, ") {
                assert_eq!(error.to_string(), message, "{}", row.name);
            } else {
                assert!(
                    matches!(
                        error,
                        GraphSpecError::Malformed(_)
                            | GraphSpecError::NoEntities
                            | GraphSpecError::TooManyEntities(_)
                            | GraphSpecError::EmptyEnum { .. }
                    ),
                    "{}: a shape error is reported as {error:?}",
                    row.name
                );
            }
            // The JSON-string entry point rejects it too.
            assert!(graph_schema_from_spec_value(&Value::String(row.spec.to_string())).is_err());
        }
    }

    #[test]
    fn spec_round_trips_through_serde_in_snake_case() {
        let row = table()
            .into_iter()
            .find(|r| r.name == "everything")
            .unwrap();
        let spec = GraphSchemaSpec::from_value(&row.spec).unwrap();
        let dumped = serde_json::to_value(&spec).unwrap();
        assert!(dumped["entities"][0].get("primary_label_field").is_some());
        assert!(!dumped.to_string().contains("primaryLabelField"));
        assert_eq!(GraphSchemaSpec::from_value(&dumped).unwrap(), spec);
        assert!(
            !spec.options.auto_type_nodes,
            "lax boolean \"no\" reads as false"
        );
    }

    #[test]
    fn root_entity_defaults_to_the_first_entity() {
        let spec = GraphSchemaSpec::from_value(&json!({
            "entities": [{"name": "Person"}, {"name": "Organization"}]
        }))
        .unwrap();
        assert_eq!(spec.root_entity().unwrap().name, "Person");
        let spec = GraphSchemaSpec {
            root: Some("Organization".into()),
            ..spec
        };
        assert_eq!(spec.root_entity().unwrap().name, "Organization");
    }

    #[test]
    fn compile_validates_a_hand_built_spec() {
        let spec = GraphSchemaSpec {
            options: GraphSchemaOptions::default(),
            root: None,
            entities: Vec::new(),
        };
        assert_eq!(
            graph_spec_to_json_schema(&spec),
            Err(GraphSpecError::NoEntities)
        );
    }

    /// Python's identifier regex is `$`-anchored, which also matches before a
    /// trailing newline; Rust rejects that name.
    #[test]
    fn a_trailing_newline_is_not_an_identifier() {
        let error =
            GraphSchemaSpec::from_value(&json!({"entities": [{"name": "Person\n"}]})).unwrap_err();
        assert!(matches!(error, GraphSpecError::InvalidEntityName(_)));
    }

    #[test]
    fn reserved_names_track_python_data_point_fields() {
        // Python `RESERVED_FIELD_NAMES` @ 37f788bd2.
        for name in [
            "belongs_to_set",
            "created_at",
            "feedback_weight",
            "id",
            "importance_weight",
            "is_type",
            "metadata",
            "ontology_uri",
            "ontology_valid",
            "source_content_hash",
            "source_node_set",
            "source_pipeline",
            "source_task",
            "source_user",
            "topological_rank",
            "type",
            "updated_at",
            "version",
        ] {
            assert!(is_reserved_field_name(name), "{name}");
        }
        // Dropped from Python's DataPoint in SDK-880.
        assert!(!is_reserved_field_name("valid_to"));
        // Reserved for a spec, as in Python, though a hand-written
        // `graph_schema` may declare it (the Rust base does not carry it).
        assert!(!DATA_POINT_FIELDS.contains(&"ontology_uri"));
    }
}
