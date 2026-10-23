//! `graph_spec_to_json_schema`: the spec-DSL compiler as a module function.

use pyo3::prelude::*;

use crate::json::{py_to_serde, serde_to_py};
use crate::sdk_error::sdk_error_to_py;

/// Compile a custom graph model written in the spec DSL — a dict, or a JSON
/// string holding one — into the JSON Schema the ``graphSchema`` cognify
/// option (compat: ``graph_model``) takes. Mirrors upstream
/// ``cognee.low_level.graph_spec_to_json_schema``.
///
/// Raises ``CogneeValidationError`` for an invalid spec, with upstream's
/// message for a rule violation.
#[pyfunction]
pub(crate) fn graph_spec_to_json_schema(
    py: Python<'_>,
    spec: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let spec = py_to_serde(&spec)?;
    let schema = cognee_bindings_common::ops::pipeline::graph_spec_to_json_schema(&spec)
        .map_err(sdk_error_to_py)?;
    serde_to_py(py, &schema)
}
