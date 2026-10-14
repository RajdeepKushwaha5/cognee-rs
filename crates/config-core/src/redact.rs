//! How configuration is shown without its credentials.
//!
//! One rule, used everywhere configuration is shown (the masked snapshot a
//! binding's `config.get()` returns, and the `Debug` output of
//! [`config_section!`](crate::config_section) structs, `Settings` and the
//! resolved adapter configs):
//!
//! * a **secret** setting that is set ([`secret_is_set`]: not `null`, not
//!   `""`) is replaced by [`REDACTED`]; an unset one shows as itself (`None`,
//!   `""`), since its emptiness reveals nothing;
//! * any **other** value is shown with the credentials embedded in it masked
//!   ([`redact_credentials`]): a URL's password (`postgres://user:pw@host`) and
//!   the value of a password-like `key=value` pair (a URL query parameter or a
//!   libpq keyword DSN such as `host=db password=pw`).

use std::borrow::Cow;
use std::fmt;

use serde_json::Value;

use crate::ConfigValue;

/// The marker that stands in for a secret value wherever configuration is
/// shown.
pub const REDACTED: &str = "***REDACTED***";

/// Whether a secret's value counts as set: anything but `null` and `""`.
/// An unset secret is shown as itself; a set one as [`REDACTED`].
#[must_use]
pub fn secret_is_set(value: &Value) -> bool {
    !(value.is_null() || value.as_str() == Some(""))
}

/// A secret's value as it may be shown: `marker` when it is set
/// ([`secret_is_set`]), the value itself otherwise.
#[must_use]
pub fn mask_secret(value: Value, marker: &str) -> Value {
    if secret_is_set(&value) {
        Value::String(marker.to_string())
    } else {
        value
    }
}

/// `Debug`-formats a secret value without revealing it: the bare [`REDACTED`]
/// marker when it is set ([`secret_is_set`] of its JSON form), its own `Debug`
/// (`None`, `""`) when it is not.
///
/// ```
/// use cognee_config_core::{REDACTED, Redacted};
///
/// assert_eq!(format!("{:?}", Redacted(&"sk-live".to_string())), REDACTED);
/// assert_eq!(format!("{:?}", Redacted(&Some("sk-live".to_string()))), REDACTED);
/// assert_eq!(format!("{:?}", Redacted(&None::<String>)), "None");
/// assert_eq!(format!("{:?}", Redacted(&String::new())), "\"\"");
/// ```
pub struct Redacted<'a, T: ?Sized>(pub &'a T);

impl<T: ConfigValue + fmt::Debug> fmt::Debug for Redacted<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if secret_is_set(&self.0.to_json()) {
            f.write_str(REDACTED)
        } else {
            fmt::Debug::fmt(self.0, f)
        }
    }
}

/// `Debug`-formats a value that is not a secret but may embed one: its own
/// `Debug` (`{:#?}` included) passed through [`redact_credentials`].
///
/// ```
/// use cognee_config_core::CredentialSafe;
///
/// let url = "postgres://cognee:hunter2@db:5432/cognee".to_string();
/// let shown = format!("{:?}", CredentialSafe(&url));
/// assert_eq!(shown, "\"postgres://cognee:***REDACTED***@db:5432/cognee\"");
/// ```
pub struct CredentialSafe<'a, T: ?Sized>(pub &'a T);

impl<T: fmt::Debug + ?Sized> fmt::Debug for CredentialSafe<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rendered = if f.alternate() {
            format!("{:#?}", self.0)
        } else {
            format!("{:?}", self.0)
        };
        f.write_str(&redact_credentials(&rendered))
    }
}

/// `key=value` names whose value is a credential (compared ignoring case).
const SENSITIVE_PARAMS: &[&str] = &[
    "password",
    "passwd",
    "pwd",
    "sslpassword",
    "secret",
    "client_secret",
    "token",
    "access_token",
    "auth_token",
    "session_token",
    "api_key",
    "apikey",
    "api-key",
];

/// Bytes that end a URL authority or a `key=value` value. `\\` and `"` cover
/// text that went through a `Debug` string escape.
const VALUE_END: &[u8] = b"&;,)}\"'\\ \t\r\n";
/// Bytes after which a `key=` may start.
const PARAM_START: &[u8] = b"?&;,({\"' \t\r\n";

/// `text` with the credentials embedded in it replaced by [`REDACTED`]:
///
/// * the password of every URL with userinfo: `scheme://user:pw@host` becomes
///   `scheme://user:***REDACTED***@host` (the user name is kept);
/// * the value of every password-like `key=value` pair (`password`, `pwd`,
///   `sslpassword`, `secret`, `token`, `api_key`, …): a URL query parameter
///   (`?sslpassword=pw`) or a libpq keyword DSN (`host=db password=pw`, also
///   `password='p w'`).
///
/// Borrowed unchanged when there is nothing to mask.
///
/// ```
/// use cognee_config_core::redact_credentials;
///
/// assert_eq!(
///     redact_credentials("postgres://u:pw@h/db?sslpassword=x&sslmode=require"),
///     "postgres://u:***REDACTED***@h/db?sslpassword=***REDACTED***&sslmode=require",
/// );
/// assert_eq!(
///     redact_credentials("host=db user=u password='p w' dbname=c"),
///     "host=db user=u password=***REDACTED*** dbname=c",
/// );
/// assert_eq!(redact_credentials("/var/lib/cognee"), "/var/lib/cognee");
/// ```
#[must_use]
pub fn redact_credentials(text: &str) -> Cow<'_, str> {
    match redact_url_passwords(text) {
        Cow::Borrowed(text) => redact_params(text),
        Cow::Owned(text) => Cow::Owned(redact_params(&text).into_owned()),
    }
}

/// Every `scheme://user:password@` password replaced by [`REDACTED`].
fn redact_url_passwords(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut search = 0;
    while let Some(found) = text[search..].find("://") {
        let start = search + found + 3;
        let end = bytes[start..]
            .iter()
            .position(|b| b"/?#".contains(b) || VALUE_END.contains(b))
            .map_or(bytes.len(), |n| start + n);
        search = end.max(start);
        let authority = &text[start..end];
        let Some(at) = authority.rfind('@') else {
            continue;
        };
        let Some(colon) = authority[..at].find(':') else {
            continue;
        };
        if colon + 1 == at {
            continue;
        }
        let buf = out.get_or_insert_with(String::new);
        buf.push_str(&text[copied..start + colon + 1]);
        buf.push_str(REDACTED);
        copied = start + at;
    }
    match out {
        Some(mut buf) => {
            buf.push_str(&text[copied..]);
            Cow::Owned(buf)
        }
        None => Cow::Borrowed(text),
    }
}

/// Every password-like `key=value` value replaced by [`REDACTED`].
fn redact_params(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary = i == 0 || PARAM_START.contains(&bytes[i - 1]);
        let name_len = if at_boundary {
            SENSITIVE_PARAMS.iter().find_map(|name| {
                let n = name.len();
                (bytes.len() > i + n
                    && bytes[i + n] == b'='
                    && bytes[i..i + n].eq_ignore_ascii_case(name.as_bytes()))
                .then_some(n)
            })
        } else {
            None
        };
        let Some(n) = name_len else {
            i += 1;
            continue;
        };
        let value_start = i + n + 1;
        let value_end = if bytes.get(value_start) == Some(&b'\'') {
            // A quoted libpq value: up to and including the closing quote.
            bytes[value_start + 1..]
                .iter()
                .position(|b| *b == b'\'')
                .map_or(bytes.len(), |p| value_start + 1 + p + 1)
        } else {
            bytes[value_start..]
                .iter()
                .position(|b| VALUE_END.contains(b))
                .map_or(bytes.len(), |p| value_start + p)
        };
        if value_end > value_start && &text[value_start..value_end] != REDACTED {
            let buf = out.get_or_insert_with(String::new);
            buf.push_str(&text[copied..value_start]);
            buf.push_str(REDACTED);
            copied = value_end;
        }
        i = value_end.max(i + 1);
    }
    match out {
        Some(mut buf) => {
            buf.push_str(&text[copied..]);
            Cow::Owned(buf)
        }
        None => Cow::Borrowed(text),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn secret_is_set_treats_null_and_empty_as_unset() {
        assert!(!secret_is_set(&json!(null)));
        assert!(!secret_is_set(&json!("")));
        assert!(secret_is_set(&json!("x")));
        assert!(secret_is_set(&json!(0)));
        assert_eq!(mask_secret(json!(""), REDACTED), json!(""));
        assert_eq!(mask_secret(json!(null), REDACTED), json!(null));
        assert_eq!(mask_secret(json!("sk"), REDACTED), json!(REDACTED));
    }

    #[test]
    fn redacts_url_passwords() {
        for (input, expected) in [
            (
                "postgres://cognee:hunter2@db:5432/cognee",
                "postgres://cognee:***REDACTED***@db:5432/cognee",
            ),
            (
                "redis://:hunter2@cache:6379/0",
                "redis://:***REDACTED***@cache:6379/0",
            ),
            (
                "a postgres://u:p@h and http://x:y@z/",
                "a postgres://u:***REDACTED***@h and http://x:***REDACTED***@z/",
            ),
            // A password containing `@` and `:`: the last `@` ends userinfo.
            (
                "postgres://u:p:a@ss@h/db",
                "postgres://u:***REDACTED***@h/db",
            ),
        ] {
            assert_eq!(redact_credentials(input), expected, "{input}");
        }
    }

    #[test]
    fn keeps_values_without_credentials() {
        for input in [
            "postgres://cognee@db/cognee",
            "http://localhost:8080/path?x=1",
            "/var/lib/cognee/system",
            "",
            "user:pass without scheme",
            "mytoken=abc",
            "postgres://u:@h/db",
        ] {
            assert!(
                matches!(redact_credentials(input), Cow::Borrowed(s) if s == input),
                "{input}"
            );
        }
    }

    #[test]
    fn redacts_password_like_params() {
        for (input, expected) in [
            (
                "postgres://h/db?sslmode=require&password=pw",
                "postgres://h/db?sslmode=require&password=***REDACTED***",
            ),
            (
                "host=db port=5432 password=pw dbname=c",
                "host=db port=5432 password=***REDACTED*** dbname=c",
            ),
            (
                "host=db PASSWORD='p w' dbname=c",
                "host=db PASSWORD=***REDACTED*** dbname=c",
            ),
            (
                "api_key=sk-1&token=t",
                "api_key=***REDACTED***&token=***REDACTED***",
            ),
        ] {
            assert_eq!(redact_credentials(input), expected, "{input}");
        }
    }

    /// Applied to `Debug` output, the escaping of a string does not let the
    /// password through.
    #[test]
    fn credential_safe_masks_debug_output() {
        let url = Some("postgres://u:hunter2@db/c?password=pw2".to_string());
        for shown in [
            format!("{:?}", CredentialSafe(&url)),
            format!("{:#?}", CredentialSafe(&url)),
        ] {
            assert!(!shown.contains("hunter2"), "{shown}");
            assert!(!shown.contains("pw2"), "{shown}");
            assert!(shown.starts_with("Some("), "{shown}");
        }
        assert_eq!(format!("{:?}", CredentialSafe(&7u32)), "7");
    }

    #[test]
    fn redacted_shows_unset_values_as_themselves() {
        assert_eq!(format!("{:?}", Redacted(&String::new())), "\"\"");
        assert_eq!(format!("{:?}", Redacted(&None::<String>)), "None");
        assert_eq!(
            format!("{:?}", Redacted(&Some(String::new()))),
            "Some(\"\")"
        );
        assert_eq!(format!("{:?}", Redacted(&"x".to_string())), REDACTED);
    }
}
