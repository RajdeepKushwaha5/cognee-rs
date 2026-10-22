#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! End-to-end `cognify()` with a runtime graph schema (SDK-1124): what a run
//! with `CognifyConfig::graph_schema` leaves in the graph store, the vector
//! store and the ownership ledger, for both chunk attachment modes, and that
//! it asks the LLM with the same prompt the default path sends.
//!
//! Python reference: `extract_graph_from_data` (custom `graph_model` branch),
//! `extract_content_graph` (prompt and temporal hints) and `add_data_points`
//! (`_walk_data_point`, `unwrap_transparent`, `chunk_attachment`).
//!
//! Offline throughout — see `rollback_harness`. The LLM answers by the shape
//! of the schema it is asked for rather than in FIFO order, so the result does
//! not depend on how the fused stage schedules its calls.

mod rollback_harness;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use cognee_cognify::fact_extraction::FactExtractor;
use cognee_cognify::{ChunkAttachment, CognifyConfig};
use cognee_graph::{EdgeData, GraphDBTrait, NodeData};
use cognee_llm::types::{GenerationOptions, GenerationResponse, Message};
use cognee_llm::{Llm, LlmError, LlmResult, MessageRole};
use cognee_models::EdgeType;
use cognee_vector::VectorDB;
use rollback_harness::{Harness, canned_graph_response, extraction_config};
use serde_json::{Value, json};
use uuid::Uuid;

/// Text without a date, so the default prompt goes out without hints.
const PLAIN_TEXT: &str = "Alice and Bob work at Acme. Alice also consults for Initech.";

/// Text with a year-less date the temporal hint finder resolves.
const DATED_TEXT: &str =
    "Acme hired Alice on 26 April 1986. The following night of 27 April, Alice started.";

fn node_id(model: &str, name: &str) -> Uuid {
    Uuid::new_v5(
        &Uuid::NAMESPACE_OID,
        format!("{model}:{}", name.to_lowercase()).as_bytes(),
    )
}

fn person_def() -> Value {
    json!({
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "metadata": {"type": "object", "default": {
                "index_fields": ["name"], "identity_fields": ["name"]
            }}
        },
        "required": ["name"]
    })
}

fn company_properties() -> Value {
    json!({
        "name": {"type": "string"},
        "employees": {"type": "array", "items": {"$ref": "#/$defs/Person"}},
        "metadata": {"type": "object", "default": {
            "index_fields": ["name"], "identity_fields": ["name"]
        }}
    })
}

/// A `Company` root with `Person` employees.
fn company_schema() -> Value {
    json!({
        "$defs": {"Person": person_def()},
        "title": "Company",
        "type": "object",
        "properties": company_properties(),
        "required": ["name", "employees"]
    })
}

fn company_answer() -> Value {
    json!({"name": "Acme", "employees": [{"name": "Alice"}, {"name": "Bob"}]})
}

/// A transparent `Roster` root holding companies: the root is never stored,
/// and its companies take its place. Alice works at both companies, so the
/// storage walk reaches her twice.
fn roster_schema() -> Value {
    json!({
        "$defs": {
            "Person": person_def(),
            "Company": {
                "title": "Company",
                "type": "object",
                "properties": company_properties(),
                "required": ["name", "employees"]
            }
        },
        "title": "Roster",
        "type": "object",
        "properties": {
            "companies": {"type": "array", "items": {"$ref": "#/$defs/Company"}},
            "metadata": {"type": "object", "default": {"transparent": true}}
        },
        "required": ["companies"]
    })
}

fn roster_answer() -> Value {
    json!({"companies": [
        {"name": "Acme", "employees": [{"name": "Alice"}, {"name": "Bob"}]},
        {"name": "Initech", "employees": [{"name": "Alice"}]}
    ]})
}

/// Answers graph extraction by the schema it is asked for — the canned
/// `KnowledgeGraph` for the default path, `custom` for any other — and records
/// every system prompt it is sent.
struct SchemaKeyedLlm {
    custom: Value,
    system_prompts: Mutex<Vec<String>>,
}

impl SchemaKeyedLlm {
    fn new(custom: Value) -> Arc<Self> {
        Arc::new(Self {
            custom,
            system_prompts: Mutex::new(Vec::new()),
        })
    }

    fn system_prompts(&self) -> Vec<String> {
        self.system_prompts.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Llm for SchemaKeyedLlm {
    async fn generate(
        &self,
        _messages: Vec<Message>,
        _options: Option<GenerationOptions>,
    ) -> LlmResult<GenerationResponse> {
        Err(LlmError::FeatureNotSupported(
            "SchemaKeyedLlm only serves structured output".to_string(),
        ))
    }

    async fn create_structured_output_with_messages_raw(
        &self,
        messages: Vec<Message>,
        json_schema: &Value,
        _options: Option<GenerationOptions>,
    ) -> LlmResult<Value> {
        let system: String = messages
            .iter()
            .filter(|message| message.role == MessageRole::System)
            .map(|message| message.content.as_str())
            .collect();
        self.system_prompts.lock().unwrap().push(system);
        let properties = json_schema.get("properties");
        if properties.and_then(|p| p.get("nodes")).is_some() {
            return serde_json::from_str(&canned_graph_response())
                .map_err(|error| LlmError::DeserializationError(error.to_string()));
        }
        if properties.and_then(|p| p.get("summary")).is_some() {
            return Ok(json!({"summary": "A summary.", "description": "A description."}));
        }
        Ok(self.custom.clone())
    }

    fn model(&self) -> &str {
        "schema-keyed"
    }
}

/// What one run left in the stores.
struct Stored {
    nodes: BTreeMap<String, NodeData>,
    edges: Vec<EdgeData>,
}

impl Stored {
    async fn of(harness: &Harness) -> Self {
        let (nodes, edges) = harness
            .graph_db
            .get_graph_data()
            .await
            .expect("read the graph store");
        Self {
            nodes: nodes.into_iter().collect(),
            edges,
        }
    }

    /// The ids of the stored nodes whose `type` is `data_type`.
    fn ids_of_type(&self, data_type: &str) -> BTreeSet<String> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.get("type") == Some(&json!(data_type)))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Every `contains` target, once per edge — a duplicate stays visible.
    fn contains_targets(&self) -> Vec<String> {
        let mut targets: Vec<String> = self
            .edges
            .iter()
            .filter(|(_, _, relationship, _)| relationship == "contains")
            .map(|(_, target, _, _)| target.clone())
            .collect();
        targets.sort();
        targets
    }

    fn field_edges(&self, field: &str) -> BTreeSet<(String, String)> {
        self.edges
            .iter()
            .filter(|(_, _, relationship, _)| relationship == field)
            .map(|(source, target, _, _)| (source.clone(), target.clone()))
            .collect()
    }
}

fn ids(pairs: &[(&str, &str)]) -> Vec<String> {
    let mut ids: Vec<String> = pairs
        .iter()
        .map(|(model, name)| node_id(model, name).to_string())
        .collect();
    ids.sort();
    ids
}

async fn run(config: &CognifyConfig, text: &str, llm: Arc<SchemaKeyedLlm>) -> (Harness, Uuid) {
    let mut harness = Harness::new().await;
    let data_id = harness.add_file(text).await;
    harness
        .run_over(config, &[data_id], llm)
        .await
        .expect("cognify succeeds");
    (harness, data_id)
}

/// The custom nodes, their field edges, their `{Model}_{field}` points, the
/// field edge's `EdgeType` point and its ledger rows — the part both
/// attachment modes share for the `Company` schema.
async fn assert_company_graph(harness: &Harness, data_id: Uuid) {
    let stored = Stored::of(harness).await;
    let acme = node_id("Company", "Acme");
    let alice = node_id("Person", "Alice");
    let bob = node_id("Person", "Bob");

    assert_eq!(
        stored.ids_of_type("Company"),
        BTreeSet::from([acme.to_string()])
    );
    assert_eq!(
        stored.ids_of_type("Person"),
        BTreeSet::from([alice.to_string(), bob.to_string()])
    );
    assert_eq!(stored.nodes[&acme.to_string()]["name"], json!("Acme"));
    assert_eq!(stored.nodes[&alice.to_string()]["name"], json!("Alice"));
    assert_eq!(
        stored.field_edges("employees"),
        BTreeSet::from([
            (acme.to_string(), alice.to_string()),
            (acme.to_string(), bob.to_string()),
        ])
    );
    assert!(
        stored.ids_of_type("Entity").is_empty(),
        "the custom path does not run the default KnowledgeGraph expansion"
    );

    let vectors = &harness.vector_db;
    let company_point = vectors
        .get_payload("Company", "name", acme)
        .expect("Company_name point");
    assert_eq!(company_point.get("text"), Some(&json!("Acme")));
    assert!(vectors.get_payload("Person", "name", alice).is_some());
    assert!(vectors.get_payload("Person", "name", bob).is_some());
    assert!(
        vectors
            .get_payload(
                "EdgeType",
                "relationship_name",
                EdgeType::deterministic_id("Acme employees Alice.")
            )
            .is_some(),
        "the field edge is indexed on its stamped text"
    );

    let owned: BTreeSet<Uuid> = harness
        .ledger_nodes()
        .await
        .into_iter()
        .filter(|row| row.data_id == data_id)
        .map(|row| row.slug)
        .collect();
    assert!(
        [acme, alice, bob].iter().all(|id| owned.contains(id)),
        "every custom node is claimed by the item: {owned:?}"
    );
    let field_rows = harness
        .ledger_edges()
        .await
        .into_iter()
        .filter(|row| row.relationship_name == "employees")
        .count();
    assert_eq!(field_rows, 2, "one ledger row per field edge");
}

#[tokio::test]
async fn direct_attachment_links_the_chunk_to_the_root_only() {
    let config = extraction_config().with_graph_schema(company_schema());
    let (harness, data_id) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(company_answer())).await;

    assert_company_graph(&harness, data_id).await;
    assert_eq!(
        Stored::of(&harness).await.contains_targets(),
        ids(&[("Company", "Acme")]),
        "omitted chunk_attachment is Python's `direct`"
    );
}

#[tokio::test]
async fn all_attachment_links_the_chunk_to_every_stored_node_once() {
    let config = extraction_config()
        .with_graph_schema(company_schema())
        .with_chunk_attachment(ChunkAttachment::All);
    let (harness, data_id) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(company_answer())).await;

    assert_company_graph(&harness, data_id).await;
    assert_eq!(
        Stored::of(&harness).await.contains_targets(),
        ids(&[("Company", "Acme"), ("Person", "Alice"), ("Person", "Bob")])
    );
}

#[tokio::test]
async fn direct_attachment_with_a_transparent_root_links_its_children() {
    let config = extraction_config().with_graph_schema(roster_schema());
    let (harness, _) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(roster_answer())).await;
    let stored = Stored::of(&harness).await;

    assert!(
        stored.ids_of_type("Roster").is_empty(),
        "a transparent root is never stored"
    );
    assert_eq!(
        stored.ids_of_type("Company").len() + stored.ids_of_type("Person").len(),
        4,
        "Acme, Initech, Alice and Bob — Alice once"
    );
    assert_eq!(
        stored.contains_targets(),
        ids(&[("Company", "Acme"), ("Company", "Initech")]),
        "the chunk links to the companies that replace the root"
    );
    assert_eq!(
        stored.field_edges("employees").len(),
        3,
        "Acme→Alice, Acme→Bob, Initech→Alice"
    );
    assert!(
        harness
            .vector_db
            .get_payload("Company", "name", node_id("Company", "Initech"))
            .is_some()
    );
}

#[tokio::test]
async fn all_attachment_with_a_transparent_root_links_every_stored_node_once() {
    let config = extraction_config()
        .with_graph_schema(roster_schema())
        .with_chunk_attachment(ChunkAttachment::All);
    let (harness, _) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(roster_answer())).await;
    let stored = Stored::of(&harness).await;

    let expected = ids(&[
        ("Company", "Acme"),
        ("Company", "Initech"),
        ("Person", "Alice"),
        ("Person", "Bob"),
    ]);
    assert_eq!(
        stored.contains_targets(),
        expected,
        "every stored node exactly once, Alice included though the walk reaches her twice"
    );
    let mut custom_nodes: Vec<String> = stored
        .ids_of_type("Company")
        .into_iter()
        .chain(stored.ids_of_type("Person"))
        .collect();
    custom_nodes.sort();
    assert_eq!(
        custom_nodes, expected,
        "and nothing is linked that is not stored"
    );
}

/// The custom path sends the default path's prompt: the vendored default
/// graph prompt, unchanged, for a chunk without dates, and the same hinted
/// prompt for a chunk with one (Python's `extract_content_graph` renders
/// `temporal_hints` whatever the response model).
#[tokio::test]
async fn the_custom_path_sends_the_default_paths_prompt() {
    for text in [PLAIN_TEXT, DATED_TEXT] {
        let default_llm = SchemaKeyedLlm::new(Value::Null);
        run(&extraction_config(), text, Arc::clone(&default_llm)).await;
        let custom_llm = SchemaKeyedLlm::new(company_answer());
        run(
            &extraction_config().with_graph_schema(company_schema()),
            text,
            Arc::clone(&custom_llm),
        )
        .await;

        let default_prompts = default_llm.system_prompts();
        assert_eq!(default_prompts.len(), 1, "one extraction call");
        assert_eq!(
            custom_llm.system_prompts(),
            default_prompts,
            "both paths send one identical prompt for {text:?}"
        );
        let prompt = &default_prompts[0];
        if text == PLAIN_TEXT {
            assert_eq!(
                prompt,
                FactExtractor::default_graph_prompt(),
                "the default prompt goes out byte for byte"
            );
        } else {
            assert!(prompt.starts_with(FactExtractor::default_graph_prompt()));
            assert!(
                prompt.contains("TEMPORAL_NORMALIZATION_HINTS") && prompt.contains("1986-04-27"),
                "the chunk's date hint is rendered: {prompt}"
            );
        }
    }
}

/// A custom extraction prompt is sent verbatim, with no hints, on the custom
/// path as on the default one.
#[tokio::test]
async fn a_custom_prompt_is_sent_verbatim_on_the_custom_path() {
    let mut config = extraction_config().with_graph_schema(company_schema());
    config.custom_extraction_prompt = Some("Extract the company.".to_string());
    let llm = SchemaKeyedLlm::new(company_answer());
    run(&config, DATED_TEXT, Arc::clone(&llm)).await;

    assert_eq!(llm.system_prompts(), ["Extract the company.".to_string()]);
}

/// Without a schema, cognify takes the default `KnowledgeGraph` path: entity
/// nodes and their relationship, the chunk linked to its entities, and no
/// custom collection.
#[tokio::test]
async fn default_cognify_without_a_schema_is_unchanged() {
    let (harness, _) = run(
        &extraction_config(),
        PLAIN_TEXT,
        SchemaKeyedLlm::new(Value::Null),
    )
    .await;
    let stored = Stored::of(&harness).await;

    let entity_names: BTreeSet<String> = stored
        .ids_of_type("Entity")
        .iter()
        .filter_map(|id| stored.nodes[id]["name"].as_str().map(str::to_string))
        .collect();
    assert_eq!(
        entity_names,
        BTreeSet::from(["alice".to_string(), "acme".to_string()])
    );
    assert_eq!(stored.field_edges("works_at").len(), 1);
    assert_eq!(
        stored.contains_targets().len(),
        2,
        "the chunk contains its two entities"
    );
    assert!(stored.ids_of_type("Company").is_empty());
    for (data_type, field) in [("Company", "name"), ("Person", "name")] {
        assert!(
            !harness
                .vector_db
                .has_collection(data_type, field)
                .await
                .expect("collection lookup"),
            "no {data_type}_{field} collection without a schema"
        );
    }
}

/// A graph model written in the spec DSL (SDK-1055) runs end to end: the
/// compiled schema is what the LLM is asked for, and the answer is stored as
/// Python's generated model would store it — one node per entity, one field
/// edge per relation, and every instance's `is_type` marker merged into one
/// `{Name}Type` node per entity.
#[tokio::test]
async fn a_spec_compiled_schema_cognifies_end_to_end() {
    let spec = cognee_cognify::GraphSchemaSpec::from_value(&json!({
        "entities": [
            {
                "name": "Person",
                "fields": [
                    {"kind": "primitive", "name": "age", "primitive_type": "number"},
                    {
                        "kind": "relation", "name": "works_at",
                        "relation": {"target_entity_name": "Organization"}
                    },
                    {
                        "kind": "relation", "name": "knows",
                        "relation": {"target_entity_name": "Person", "cardinality": "many"}
                    }
                ]
            },
            {"name": "Organization"}
        ]
    }))
    .expect("valid spec");
    let schema = cognee_cognify::graph_spec_to_json_schema(&spec).expect("spec compiles");
    let answer = json!({
        "name": "Ada",
        "is_type": {},
        "age": 36,
        "works_at": {"name": "Analytical Engines", "is_type": {}},
        "knows": [{"name": "Charles", "is_type": {"name": "Person"}}]
    });
    let config = extraction_config().with_graph_schema(schema);
    let (harness, _) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(answer)).await;
    let stored = Stored::of(&harness).await;

    // The ids Python's `graph_model_from_spec` model assigns to this answer.
    let ada = "19602cf7-f83a-5764-884b-09504eb8f348";
    let charles = "e69f3bc4-5260-54cc-9692-8194cedd27a7";
    let engines = "2f9555c5-730a-5e47-b896-0e54caf69083";
    let person_type = "b4404d21-cd36-5bc7-9b83-8e08176c280a";
    let organization_type = "b4323b15-10bc-5a05-b72c-b4dcaa343dd9";
    let set = |ids: &[&str]| ids.iter().map(|id| id.to_string()).collect::<BTreeSet<_>>();
    let edge = |source: &str, target: &str| (source.to_string(), target.to_string());

    assert_eq!(stored.ids_of_type("Person"), set(&[ada, charles]));
    assert_eq!(stored.ids_of_type("Organization"), set(&[engines]));
    assert_eq!(
        stored.ids_of_type("PersonType"),
        set(&[person_type]),
        "both people's markers merge into one PersonType node"
    );
    assert_eq!(
        stored.ids_of_type("OrganizationType"),
        set(&[organization_type])
    );
    assert_eq!(
        stored.field_edges("works_at"),
        BTreeSet::from([edge(ada, engines)])
    );
    assert_eq!(
        stored.field_edges("knows"),
        BTreeSet::from([edge(ada, charles)])
    );
    assert_eq!(
        stored.field_edges("is_type"),
        BTreeSet::from([
            edge(ada, person_type),
            edge(charles, person_type),
            edge(engines, organization_type),
        ])
    );
    // A `number` field is a Python float.
    assert_eq!(stored.nodes[ada].get("age"), Some(&json!(36.0)));
}

/// A `Person` with a FromIdentity `employer` (SDK-1055) next to a root that
/// lists the companies in full.
fn directory_schema(employer: Value) -> Value {
    json!({
        "$defs": {
            "Company": {
                "title": "Company",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "founded": {"type": "integer", "default": 0},
                    "metadata": {"type": "object", "default": {
                        "index_fields": ["name"], "identity_fields": ["name"]
                    }}
                },
                "required": ["name"]
            },
            "Person": {
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "employer": employer,
                    "metadata": {"type": "object", "default": {
                        "index_fields": ["name"], "identity_fields": ["name"]
                    }}
                },
                "required": ["name"]
            }
        },
        "title": "Directory",
        "type": "object",
        "properties": {
            "companies": {"type": "array", "items": {"$ref": "#/$defs/Company"}},
            "people": {"type": "array", "items": {"$ref": "#/$defs/Person"}},
            "metadata": {"type": "object", "default": {"transparent": true}}
        },
        "required": ["companies", "people"]
    })
}

/// The LLM answers a FromIdentity field with the company's name, and the
/// stored edge points at the company node the same answer lists in full.
#[tokio::test]
async fn from_identity_answers_reference_existing_nodes() {
    let schema = directory_schema(json!({
        "anyOf": [{"$ref": "#/$defs/Company"}, {"type": "null"}],
        "default": null,
        "x-cognee-from-identity": true
    }));
    let answer = json!({
        "companies": [{"name": "Acme", "founded": 1999}],
        "people": [{"name": "Alice", "employer": "Acme"}, {"name": "Bob", "employer": null}]
    });
    let config = extraction_config().with_graph_schema(schema);
    let (harness, _) = run(&config, PLAIN_TEXT, SchemaKeyedLlm::new(answer)).await;
    let stored = Stored::of(&harness).await;

    let acme = node_id("Company", "Acme").to_string();
    assert_eq!(
        stored.ids_of_type("Company"),
        BTreeSet::from([acme.clone()])
    );
    assert_eq!(
        stored.field_edges("employer"),
        BTreeSet::from([(node_id("Person", "Alice").to_string(), acme.clone())])
    );
    assert_eq!(
        stored.nodes[&acme].get("founded"),
        Some(&json!(1999)),
        "the full answer, reached first, keeps its properties"
    );
}

/// A FromIdentity reference Python would refuse fails the run before any
/// LLM call.
#[tokio::test]
async fn an_invalid_from_identity_reference_fails_before_any_llm_call() {
    let schema = directory_schema(json!({
        "type": "string",
        "x-cognee-from-identity": true
    }));
    let llm = SchemaKeyedLlm::new(json!({}));
    let mut harness = Harness::new().await;
    let data_id = harness.add_file(PLAIN_TEXT).await;
    let error = harness
        .run_over(
            &extraction_config().with_graph_schema(schema),
            &[data_id],
            llm.clone(),
        )
        .await
        .expect_err("an invalid FromIdentity reference fails the run");
    assert!(
        error
            .to_string()
            .contains("FromIdentity reference must name a model"),
        "{error}"
    );
    assert!(llm.system_prompts().is_empty(), "no LLM call was made");
}

/// A reference stored in a later run is written over the node an earlier run
/// stored, with the target's defaults: Python's Ladybug adapter replaces the
/// whole `properties` blob on `MERGE ... ON MATCH SET`
/// (`ladybug/adapter.py` `add_nodes`), and so does Rust's. Pinned so a change
/// to these semantics is a deliberate one.
#[tokio::test]
async fn a_later_reference_overwrites_the_stored_node_as_in_python() {
    let schema = directory_schema(json!({
        "anyOf": [{"$ref": "#/$defs/Company"}, {"type": "null"}],
        "default": null,
        "x-cognee-from-identity": true
    }));
    let config = extraction_config().with_graph_schema(schema);
    let mut harness = Harness::new().await;
    let first = harness.add_file(PLAIN_TEXT).await;
    let second = harness.add_file(DATED_TEXT).await;
    let acme = node_id("Company", "Acme").to_string();

    let full = json!({"companies": [{"name": "Acme", "founded": 1999}], "people": []});
    harness
        .run_over(&config, &[first], SchemaKeyedLlm::new(full))
        .await
        .expect("first run");
    assert_eq!(
        Stored::of(&harness).await.nodes[&acme].get("founded"),
        Some(&json!(1999))
    );

    let reference = json!({"companies": [], "people": [{"name": "Alice", "employer": "Acme"}]});
    harness
        .run_over(&config, &[second], SchemaKeyedLlm::new(reference))
        .await
        .expect("second run");
    let stored = Stored::of(&harness).await;
    assert_eq!(
        stored.field_edges("employer"),
        BTreeSet::from([(node_id("Person", "Alice").to_string(), acme.clone())]),
        "the reference resolves to the stored node"
    );
    assert_eq!(
        stored.nodes[&acme].get("founded"),
        Some(&json!(0)),
        "the reference's defaults replace the stored properties"
    );
}
