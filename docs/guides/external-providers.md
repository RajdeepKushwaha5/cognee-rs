# External providers

**cognee-rust is designed to be extended from outside this repository.** Any
vector store, graph store or LLM can be plugged in by a crate that does not
live here and that this repository has never heard of. You implement a factory
trait, register it, and select it by provider id in configuration. No fork, no
patch, no upstream PR.

This is a design commitment, not a convenience: adding a provider must never
require a change to cognee-rust. If you hit something that does, it is a bug in
this repository — please report it.

## The seam

Three pieces, all in `cognee-components` and re-exported from `cognee`:

| Piece | Role |
|---|---|
| `VectorDbFactory` / `GraphDbFactory` / `LlmFactory` | implement one for your backend; `provider()` returns the id you want to be selected by |
| `BackendBuildContext` | the resolved, environment-free inputs your `build()` receives |
| `ComponentRegistry` | the provider-id → factory map; `register_vector` / `register_graph` / `register_llm` add yours |

Your factory is looked up by the same path as every built-in one
(`ComponentRegistry::build_vector` / `build_graph` / `build_llm`), so it gets
identical treatment — including the LLM mock-override and recording wrappers.
An `LlmFactory` whose provider needs no API key overrides `requires_api_key` to
return `false`; otherwise, with `LLM_API_KEY` empty, `build()` is not called and
the LLM is unavailable.

## Registering one

```rust
use std::sync::Arc;
use async_trait::async_trait;
use cognee::{BackendBuildContext, ComponentError, ComponentManager, ComponentRegistry,
             ConfigManager, VectorDbFactory};
use cognee_vector::VectorDB;

struct MyVectorFactory;

#[async_trait]
impl VectorDbFactory for MyVectorFactory {
    fn provider(&self) -> &str {
        "my-store"
    }

    async fn build(&self, ctx: &BackendBuildContext) -> Result<Arc<dyn VectorDB>, ComponentError> {
        let adapter = MyAdapter::new(&ctx.vector_db_url, ctx.embedding_dimensions)
            .await
            .map_err(|e| ComponentError::VectorDb(format!("my-store init failed: {e}")))?;
        Ok(Arc::new(adapter))
    }
}

let mut registry = ComponentRegistry::with_builtins();
registry.register_vector(Arc::new(MyVectorFactory));

let manager = ComponentManager::with_registry(ConfigManager::from_env(), registry);
```

Then select it:

```sh
VECTOR_DB_PROVIDER=my-store
```

Provider ids are matched lowercase. Registering under an id that a built-in
already uses **replaces** the built-in — that is supported, and is how you
substitute your own `lancedb` or `pgvector` implementation.

In the bindings, use `HandleState::from_settings_with_registry`.

## Postgres-backed providers

If your backend sits on PostgreSQL, you do **not** need to assemble a
connection URL from the `*_HOST` / `*_PORT` / `*_NAME` / `*_USERNAME` /
`*_PASSWORD` settings yourself, and you must not re-read the environment to do
it. The context already carries the resolution:

```rust
async fn build(&self, ctx: &BackendBuildContext) -> Result<Arc<dyn VectorDB>, ComponentError> {
    let url = match ctx.vector_postgres_url.as_ref() {
        Some(Ok(url)) => url,
        // Resolution was attempted and failed — restate the real cause
        // (e.g. missing credentials) rather than a generic message.
        Some(Err(cause)) => return Err(ComponentError::Config(cause.clone())),
        None => {
            return Err(ComponentError::Config(
                "vector_db_provider=my-store requires a resolved PostgreSQL URL".into(),
            ));
        }
    };
    // …
}
```

`graph_postgres_url` is the same for graph backends, resolved from the
`GRAPH_DATABASE_*` settings with a fallback to the relational `DB_*` set when
the graph-specific credentials are incomplete (the single-shared-Postgres
layout).

Both fields are populated **for every provider id**, including yours. They are
not a signal that the selected provider is Postgres-backed — a provider with
nothing to do with Postgres simply ignores them. The three states are:

- `Some(Ok(url))` — a URL was **assembled from configuration**; use it.
- `Some(Err(cause))` — assembly was attempted and failed; `cause` names the
  reason, surface it.
- `None` — the caller attempted no resolution. `Settings::backend_context`
  always attempts one, so this means a hand-built context; treat it the way the
  built-in factories do and return a clear configuration error.

> **`Ok` means "a URL was assembled", not "a Postgres server exists."** When no
> Postgres-specific settings are present, assembly falls back to the relational
> `DB_*` configuration, whose own defaults are `localhost:5432/cognee_db` — so a
> deployment that never configured Postgres still yields a syntactically valid
> URL pointing at whatever happens to be listening on the local port. This is
> long-standing behaviour that built-in `pgvector` shares. Fail with a message
> that names the setting the operator should have set, rather than letting a
> connection error to an unrelated local database stand as the diagnosis.

You do not need the `pgvector` or `pggraph` cargo features enabled to receive
these. Those features compile cognee's *own* Postgres adapters and are
unrelated to yours.

## What the context gives you

| Field | Use |
|---|---|
| `vector_db_url` / `graph_file_path` | the raw configured URL or path, for backends that take one verbatim |
| `vector_postgres_url` / `graph_postgres_url` | assembled Postgres URL, per above |
| `relational_db_url` | the fully-resolved relational store URL |
| `embedding_dimensions` | vector width, for backends that must declare it at table-creation time |
| `data_root_directory` / `system_root_directory` | derive on-disk paths from these, never from the environment |
| `embedding` / `llm` | resolved embedding and LLM inputs |

The contract is that **all** environment reading and config-specific assembly
happens when a caller lowers its config into a `BackendBuildContext`. Factories
are pure with respect to it: given the same context they build the same
backend. Do not read environment variables in `build()` — a caller that
constructed the context by hand (tests, embedded uses, the HTTP server) would
silently get different behaviour.

## If the seam does not fit

Two things make a provider unreachable, and both are bugs here rather than
things to work around:

- **A resolved input you need is only populated for built-in provider ids.**
  That is what the Postgres URLs used to do. The fix is to populate it for every
  provider, as they now are.
- **A provider id is rejected before the registry is consulted.** Nothing should
  validate a provider id against a fixed list; the registry's own lookup is the
  only authority, and its error already lists what is registered.

Please open an issue rather than forking.

## Where this lives in the code

- `crates/components/src/traits.rs` — the factory traits.
- `crates/components/src/registry.rs` — `ComponentRegistry`, `with_builtins`,
  the `register_*` methods and the shared `build_*` path.
- `crates/components/src/context.rs` — `BackendBuildContext` and the
  construction contract.
- `crates/components/src/builtins/` — the built-in factories, which are worked
  examples of everything above.
- `crates/lib/src/config.rs` — `Settings::backend_context`, where config is
  lowered into a context.
