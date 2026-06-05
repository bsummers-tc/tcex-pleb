# tcex-pleb

A [TcEx](https://github.com/ThreatConnect-Inc/tcex) submodule providing shared low-level
primitives used across all TcEx projects: caching descriptors, threading utilities, a pub/sub
event bus, proxy helpers, and an extended JMESPath function library.

## Overview

"Pleb" (short for *plebeian* — common, shared) is the foundation layer. It has no dependency on
any other TcEx submodule except `tcex/util` for a few datetime helpers in the JMESPath functions.
Everything in this submodule is imported directly from individual modules — `__init__.py` exports
nothing at the package level.

## Module Reference

### Caching Descriptors

#### `cached_property`

Extends `functools.cached_property` with a class-level `instances` registry and a `_reset()`
static method that clears the cached value on every tracked instance at once. Used wherever a
cached property needs to be invalidatable globally (e.g., between test runs or when the framework
reinitializes after a context switch).

#### `cached_property_filesystem`

A two-layer cache descriptor (memory + filesystem) with a configurable TTL (default 24 hours).
The primary use case is **forked processes**: the parent process computes the value and writes it
to a pickle file; forked children read that file on their first access instead of making a
redundant API call.

Access order:

1. **Memory** — if the value is cached in the instance dict and within TTL, return immediately.
2. **Filesystem** — if a pickle file exists at `{tempdir}/tcex_cache/{ClassName}_{attr}.pkl` and
   is within TTL, load it into memory and return.
3. **Compute** — call the wrapped function, write the result to both memory and the filesystem
   atomically (write-to-temp then rename), and return.

`_reset()` clears all in-memory cache entries and deletes all `*.pkl` files under the cache
directory. Uses weak references so tracked instances can be garbage collected normally.

#### `scoped_property`

A thread-and-process-local property descriptor. Each thread gets its own independently computed
value. Forked processes also get a fresh value because the PID is checked on every access —
if the current PID differs from the stored PID the factory is re-invoked. `_reset()` replaces
the internal `threading.local` object, effectively clearing all thread-local caches.

---

### Singleton and Null-Object Patterns

#### `Singleton`

Thread-safe metaclass (`type` subclass). Classes that use `metaclass=Singleton` return the same
instance on every construction call. The instance cache is protected by a `threading.Lock`.
Used by `LayoutJson`, `JobJson`, `Event`, `NoneModel`, and others throughout the framework.

#### `NoneModel`

A `Singleton` that returns `None` for every attribute access. Used as a null-object replacement
for optional model references, avoiding `None` guard checks at call sites.

---

### Threading

#### `ExceptionThread`

A `threading.Thread` subclass that captures any uncaught exception raised inside `run()` into
`self.exception`. After calling `join()`, callers can inspect `thread.exception` to detect and
re-raise thread failures. Also logs the exception before re-raising so it appears in the App log.

#### `Event`

A `Singleton` pub/sub event bus. Channels are arbitrary strings; any number of callbacks can
subscribe to a channel and all are invoked when `send()` is called.

```python
event = Event()
event.subscribe('my.channel', lambda **kw: print(kw))
event.send('my.channel', data='hello')
event.unsubscribe('my.channel', callback)
```

---

### Utilities

#### `proxies()`

Formats proxy connection details into a `requests`-compatible dict
(`{'http': '...', 'https': '...'}`). Accepts an optional `Sensitive` password so the value is
never inadvertently logged or stringified. Returns an empty dict when `proxy_host` or
`proxy_port` is `None`.

---

### JMESPath Extensions

The `jmespath_custom` module provides a fully extended JMESPath function library built on the
standard `jmespath` package.

#### `jmespath_options()` / `TcFunctions`

`jmespath_options()` returns a `jmespath.Options` instance pre-configured with `TcFunctions` and
`collections.OrderedDict` as the dict class (preserving key order). Pass this to
`jmespath.search()` to access all custom functions.

`TcFunctions` composes seven mixin classes (plus the standard `jmespath.functions.Functions`) via
multiple inheritance. It also overrides `_validate_arguments` to support optional and variadic
function parameters, which the standard library does not handle.

#### `JmespathFunctionsBase`

Shared private helpers for expression-reference operations used by the mixin classes:
`_get_expression_key()` and `_update_expression_parent()`. Not intended for direct use.

#### Function Mixins

| Mixin | Functions |
|---|---|
| `ArrayFunctionsMixin` | `array_join`, `cartesian`, `chunk`, `compact`, `dedup`, `delete`, `difference`, `expand`, `fill`, `flatten_list`, `group_adjacent`, `group_by`, `index_array`, `intersect`, `null_leaf`, `symmetric_difference`, `union`, `zip`, `zip_merge`, `zip_to_objects` |
| `StringFunctionsMixin` | `base64_decode`, `base64_encode`, `capitalize`, `decode_url`, `defang`, `encode_url`, `equal_fold`, `lower`, `pattern_match`, `refang`, `regex`, `regex_match`, `regex_replace`, `replace`, `split`, `string_interpolate`, `trim`, `upper` |
| `DateTimeFunctionsMixin` | `datetime_format`, `datetime_now`, `datetime_now_utc`, `datetime_to_epoch`, `format_datetime` *(deprecated)* |
| `MathFunctionsMixin` | `add`, `divide`, `mod`, `multiply`, `power`, `rand`, `subtract`, `to_int` |
| `ObjectFunctionsMixin` | `exclude_keys`, `exclude_values`, `include_keys`, `json_parse`, `json_stringify`, `merge`, `to_key_value_array`, `yaml_parse` |
| `ControlFunctionsMixin` | `has_value`, `in`, `ternary`, `type` |
| `CryptoFunctionsMixin` | `semver_compare`, `uuid`, `uuid5` |

---

## Project Structure Note — No `pyproject.toml` or `.pre-commit-config.yaml`

This submodule intentionally ships **without** a `pyproject.toml` or `.pre-commit-config.yaml`.
All linting (`ruff`), type-checking (`ty`), and pre-commit hooks are configured in the **parent
projects** (`tcex`, `tcex-app-testing`, `tcex-cli`), each of which scans this submodule as part
of its own workspace. Running `pre-commit run --all-files` or `ty check` from the parent repo
root covers this code automatically — there is no need for (and no benefit to) duplicating that
configuration here.

## Used By

- [tcex](https://github.com/ThreatConnect-Inc/tcex) — caching descriptors, threading, event bus, JMESPath
- [tcex-app-testing](https://github.com/ThreatConnect-Inc/tcex-app-testing) — `cached_property._reset()` between test runs
- [tcex-cli](https://github.com/ThreatConnect-Inc/tcex-cli) — JMESPath extensions, proxy helpers

## License

Apache 2.0 — see [LICENSE](LICENSE).
