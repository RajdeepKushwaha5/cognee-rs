//! Attaching to the interpreter from threads Python does not own.
//!
//! In PyO3 0.29, `Python::attach` is documented to panic when the interpreter
//! cannot be attached to — above all while it is finalizing. Most attaches in this
//! crate run on tokio worker threads (the bodies of `future_into_py`, pipeline
//! task closures, watcher hooks), which keep running while a Python process
//! exits — e.g. a background `PipelineRunHandle` run still in flight at
//! interpreter shutdown. A panic there unwinds through a tokio task instead
//! of failing cleanly, so those call sites go through the helpers below,
//! which use `Python::try_attach` and never panic.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const NOT_ATTACHABLE: &str =
    "the Python interpreter is not available (finalizing or not initialized)";

/// Attach and run `f`, or return a `RuntimeError` if the interpreter cannot be
/// attached to. For call sites that must produce a result.
///
/// Building the `PyErr` does not need the interpreter: `new_err` is lazy and
/// only materialises the exception object if something later attaches.
pub(crate) fn attach_or_err<F, R>(f: F) -> PyResult<R>
where
    F: for<'py> FnOnce(Python<'py>) -> PyResult<R>,
{
    Python::try_attach(f).unwrap_or_else(|| Err(PyRuntimeError::new_err(NOT_ATTACHABLE)))
}

/// Attach and run `f` if possible; otherwise skip it and log at debug level.
/// For best-effort callbacks (watcher hooks) whose outcome nobody awaits.
pub(crate) fn attach_best_effort<F>(what: &str, f: F)
where
    F: for<'py> FnOnce(Python<'py>),
{
    if Python::try_attach(f).is_none() {
        tracing::debug!(
            callback = what,
            "{NOT_ATTACHABLE}; skipping Python callback"
        );
    }
}
