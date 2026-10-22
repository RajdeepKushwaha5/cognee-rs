#!/usr/bin/env python3
"""Regenerate `graph_spec_table.json`, the ground truth for the spec DSL in
`crates/cognify/src/graph_spec.rs`.

Each row is a spec, run through cognee's own `graph_spec_to_json_schema`
(`cognee/modules/graph_models`, cognee @ 37f788bd2). A spec Python accepts
records the compiled schema; a spec Python rejects records `"error"` with the
first line of Pydantic's message, so the Rust test can check it rejects the
same documents. Accepted specs also record whether `graph_model_from_spec`
builds a model from them (a lowercase root entity compiles but cannot be
built, as datamodel-code-generator renames the class).

Run with a Python that has cognee installed:

    python3 generate.py > graph_spec_table.json

The Rust test never derives an expectation from Rust code; it replays this file.
"""

import json
import logging
import sys

logging.disable(logging.CRITICAL)

from pydantic import ValidationError  # noqa: E402

from cognee.modules.graph_models import (  # noqa: E402
    graph_model_from_spec,
    graph_spec_to_json_schema,
)

PEOPLE = {
    "root": "Person",
    "entities": [
        {
            "name": "Person",
            "description": "A person.",
            "fields": [
                {"kind": "primitive", "name": "age", "primitive_type": "number"},
                {"kind": "enum", "name": "mood", "enum_values": ["happy", "grumpy"]},
                {
                    "kind": "relation",
                    "name": "works_at",
                    "relation": {"target_entity_name": "Organization", "cardinality": "one"},
                },
                {
                    "kind": "relation",
                    "name": "knows",
                    "relation": {"target_entity_name": "Person", "cardinality": "many"},
                },
            ],
        },
        {"name": "Organization", "identity_fields": ["name"], "fields": []},
    ],
}

# Every field kind and option, camelCase aliases, Pydantic lax booleans,
# descriptions (an empty one is dropped), required flags (ignored on a
# relation), an explicit `name`, explicit index/identity/label fields,
# accepted-but-ignored inverse and options, and a non-first root.
EVERYTHING = {
    "options": {"autoTypeNodes": "no"},
    "root": "Project",
    "entities": [
        {
            "name": "Team",
            "description": "",
            "primaryLabelField": "slug",
            "indexFields": [],
            "identityFields": [],
            "fields": [
                {"kind": "primitive", "name": "slug", "required": "yes", "description": ""},
                {
                    "kind": "relation",
                    "name": "members",
                    "required": True,
                    "relation": {
                        "targetEntityName": "Person",
                        "cardinality": "many",
                        "inverse": {"enabled": 0, "name": "member_of", "cardinality": "one"},
                    },
                },
            ],
        },
        {
            "name": "Person",
            "fields": [
                {"kind": "primitive", "name": "name", "primitive_type": "string", "required": 1},
                {"kind": "primitive", "name": "born", "primitiveType": "date", "required": True},
                {"kind": "primitive", "name": "active", "primitive_type": "boolean"},
                {"kind": "primitive", "name": "height", "primitive_type": "number"},
                {
                    "kind": "enum",
                    "name": "role",
                    "enumValues": ["lead", "dev"],
                    "required": "on",
                    "description": "Role in the team.",
                },
            ],
            "index_fields": ["name", "role"],
            "identity_fields": ["name", "born"],
        },
        {
            "name": "Project",
            "description": "A project and its team.",
            "fields": [
                {
                    "kind": "relation",
                    "name": "team",
                    "description": "Who works on it.",
                    "relation": {"target_entity_name": "Team"},
                },
                {
                    "kind": "relation",
                    "name": "parent",
                    "relation": {"target_entity_name": "Project", "inverse": None},
                },
            ],
        },
    ],
}

ACCEPTED = {
    "people": PEOPLE,
    "everything": EVERYTHING,
    "single_entity_defaults": {"entities": [{"name": "Thing"}]},
    "lowercase_root": {"entities": [{"name": "thing", "fields": []}]},
    "underscored_names": {
        "entities": [
            {
                "name": "Org_unit",
                "fields": [{"kind": "primitive", "name": "cost_center_2"}],
            },
            {"name": "Type"},
        ]
    },
}


def bad(**entity):
    return {"entities": [dict({"name": "Person"}, **entity)]}


def prim(name, **extra):
    return dict({"kind": "primitive", "name": name}, **extra)


REJECTED = {
    "no_entities": {"entities": []},
    "missing_entities": {},
    "too_many_entities": {"entities": [{"name": f"E{i}"} for i in range(51)]},
    "unknown_top_level_key": {"entities": [{"name": "P"}], "surprise": 1},
    "unknown_entity_key": bad(surprise=True),
    "unknown_field_key": bad(fields=[prim("age", surprise=True)]),
    "relation_key_on_primitive": bad(
        fields=[prim("age", relation={"target_entity_name": "Person"})]
    ),
    "missing_kind": bad(fields=[{"name": "age"}]),
    "unknown_kind": bad(fields=[{"kind": "list", "name": "age"}]),
    "unknown_primitive_type": bad(fields=[prim("age", primitive_type="integer")]),
    "null_primitive_type": bad(fields=[prim("age", primitive_type=None)]),
    "null_required": bad(fields=[prim("age", required=None)]),
    "non_bool_required": bad(fields=[prim("age", required="maybe")]),
    "two_required": bad(fields=[prim("age", required=2)]),
    "null_fields": bad(fields=None),
    "null_options": {"options": None, "entities": [{"name": "P"}]},
    "empty_enum": bad(fields=[{"kind": "enum", "name": "e", "enum_values": []}]),
    "non_string_enum_value": bad(fields=[{"kind": "enum", "name": "e", "enum_values": [1]}]),
    "unknown_cardinality": bad(
        fields=[
            {
                "kind": "relation",
                "name": "r",
                "relation": {"target_entity_name": "Person", "cardinality": "few"},
            }
        ]
    ),
    "snake_and_camel_alias": bad(index_fields=["name"], indexFields=["name"]),
    "duplicate_entity": {"entities": [{"name": "P"}, {"name": "P"}]},
    "entity_name_injection": {"entities": [{"name": '"; import os'}]},
    "entity_name_dunder": {"entities": [{"name": "__class__"}]},
    "entity_name_digit": {"entities": [{"name": "1st"}]},
    "entity_name_space": {"entities": [{"name": "has space"}]},
    "entity_name_dash": {"entities": [{"name": "has-dash"}]},
    "entity_name_empty": {"entities": [{"name": ""}]},
    "entity_name_unicode": {"entities": [{"name": "Café"}]},
    "type_marker_collision": {"entities": [{"name": "Person"}, {"name": "PersonType"}]},
    "unknown_root": {"root": "Ghost", "entities": [{"name": "P"}]},
    "too_many_fields": bad(fields=[prim(f"f{i}") for i in range(41)]),
    "duplicate_field": bad(fields=[prim("age"), prim("age")]),
    "field_name_injection": bad(fields=[prim('"; import os')]),
    "field_name_space": bad(fields=[prim("a b")]),
    "reserved_id": bad(fields=[prim("id")]),
    "reserved_metadata": bad(fields=[prim("metadata")]),
    "reserved_is_type": bad(fields=[prim("is_type")]),
    "reserved_source_pipeline": bad(fields=[prim("source_pipeline")]),
    "reserved_ontology_uri": bad(fields=[prim("ontology_uri")]),
    "name_as_number": bad(fields=[prim("name", primitive_type="number")]),
    "name_as_enum": bad(fields=[{"kind": "enum", "name": "name", "enum_values": ["a"]}]),
    "unknown_relation_target": bad(
        fields=[{"kind": "relation", "name": "r", "relation": {"target_entity_name": "Nowhere"}}]
    ),
    "index_field_is_relation": bad(
        index_fields=["r"],
        fields=[{"kind": "relation", "name": "r", "relation": {"target_entity_name": "Person"}}],
    ),
    "identity_field_undeclared": bad(identity_fields=["nickname"]),
    "label_field_undeclared": bad(primary_label_field="nickname"),
}


def first_error_line(error):
    # Line 0 counts the errors. A field error puts its location on line 1 and
    # the message on line 2; a model-validator error has no location, so its
    # message ("  Value error, ...") is line 1.
    lines = str(error).splitlines()
    message = lines[1] if lines[1].startswith(" ") else lines[2]
    return message.strip().split(" [type=")[0]


def main():
    rows = []
    for name, spec in ACCEPTED.items():
        schema = graph_spec_to_json_schema(spec)
        try:
            graph_model_from_spec(spec)
            buildable = True
        except Exception:  # noqa: BLE001 - any failure means "cannot be built"
            buildable = False
        rows.append({"name": name, "spec": spec, "schema": schema, "buildable": buildable})
    for name, spec in REJECTED.items():
        try:
            graph_spec_to_json_schema(spec)
        except ValidationError as error:
            rows.append({"name": name, "spec": spec, "error": first_error_line(error)})
            continue
        sys.exit(f"{name}: Python accepted a spec listed as rejected")
    json.dump(rows, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
