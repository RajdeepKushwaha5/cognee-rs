#![cfg(feature = "pdf-pdfium")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
//! Drives the PDF loader over a *real* PDF and pins the extracted text.
//!
//! Until this file, no test anywhere in the repository fed real PDF bytes to a
//! real extractor. Every PDF in the suite was a synthesized `%PDF-1.x` header:
//! two in `pipeline.rs` that only ever reach MIME classification, and
//! `pdf_pdfium_cold_cache.rs`'s, which is deliberately barely-a-PDF because
//! that test is about the process aborting *before* extraction is reached.
//!
//! So `PdfLoader::extract` returning correct text had no coverage at all, and
//! neither did the page-header format that matches Python's `pypdf_loader.py` —
//! `format_pages` was tested only against hand-written `Vec`s, never against
//! what a PDF engine actually hands it.
//!
//! # Why this is gated on `pdf-pdfium`
//!
//! `pdf-pdfium` is in `crates/lib`'s default feature list, so a workspace test
//! run unifies it on and this file executes — and because `loaders::pdf` gives
//! pdfium priority when both backends are compiled in, pdfium is what a
//! workspace lane actually exercises.
//!
//! Note that `pdf-pure-rust` is *also* on in a workspace build, contrary to
//! what one might assume from `cognee-ingestion`'s own `default = []`: the
//! `python` crate is a root workspace member (`Cargo.toml`) and ships
//! `pdf-pure-rust` in its defaults, so feature unification turns it on for
//! every `--workspace` command. VERIFIED with `cargo metadata`. That is why
//! `pdf_pure_rust_fixture_extraction.rs` needs a lane of its own to be worth
//! anything — see its module docs.
//!
//! # When this test skips — and when it must not
//!
//! The pdfium backend needs `libpdfium`, which `pdfium-auto` downloads on first
//! use. On a developer machine that cannot obtain it — no network, a blocked
//! proxy, an unsupported target triple — this test returns instead of failing.
//!
//! **In CI it fails instead.** A skip here is not visible: cargo captures the
//! output of a *passing* test, so a bare `test ... ok` is all a normal
//! `cargo test` prints, and a lane could go green having never parsed a PDF —
//! which is precisely the green-zero failure this file and its `[[test]]`
//! entry exist to close. Keying the hard failure on `CI` (set by GitHub
//! Actions) keeps the local convenience without letting the guarantee evaporate
//! where it is load-bearing. The repo's own runner passes `--no-capture`
//! (`scripts/run_tests_with_openai.sh`), so the notice is legible there too.
//!
//! If a CI environment ever genuinely cannot download it, provision the library
//! and point `PDFIUM_LIB_PATH` at it rather than relaxing this.
//!
//! The escape hatch is otherwise deliberately narrow: it fires *only* on the
//! library-resolution error, and it loses no coverage, because that path is
//! itself what `pdf_pdfium_cold_cache.rs` pins. A parse failure, a bind
//! failure, or the wrong text still fail the test.
#![allow(
    clippy::disallowed_methods,
    reason = "tests drive configuration through the process env"
)]

use cognee_ingestion::loaders::{DocumentLoader, LoaderError, LoaderOutput, pdf::PdfLoader};
use encrypted::ENCRYPTED_EMPTY_USER_PASSWORD;
use support::{ENCRYPT_NULL_TEXT, document, encrypt_null_pdf, pdf_of};

/// A two-page, uncompressed, base-14-font PDF authored in this repository.
/// See `fixtures/pdf/README.md` for provenance and how to regenerate it.
const SAMPLE_PDF: &[u8] = include_bytes!("fixtures/pdf/sample.pdf");

#[path = "fixtures/pdf/encrypted.rs"]
mod encrypted;

#[path = "fixtures/pdf/support.rs"]
mod support;

/// The loader's expected output, CRLF-normalised — see [`normalise`].
const EXPECTED: &str = include_str!("fixtures/pdf/sample.expected.txt");

/// The *resolution* error's own prefix (`loaders::pdf::pdfium::ensure_library`),
/// and deliberately not the word "PDFium": `extract_text`'s bind failure reads
/// "Failed to load PDFium library: …", and that one is a genuine failure that
/// must not be swallowed as an environment skip.
const LIBRARY_UNOBTAINABLE: &str = "Failed to obtain the PDFium library";

/// Whether this is an automated run that must not skip. GitHub Actions sets
/// `CI=true`; some local tooling exports `CI=false` or an empty `CI`, and
/// treating merely-present as true would hard-fail those developers for no
/// reason, so the value is read rather than just its existence.
fn in_ci() -> bool {
    match std::env::var("CI") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false"),
        Err(_) => false,
    }
}

/// PDFium reports intra-page line breaks as `\r\n`, while the `Page N:`
/// separators come from `format_pages` as `\n`. Committing that mixture as a
/// text fixture invites an editor or a `.gitattributes` rule to rewrite it, so
/// the fixture is stored LF-only and both sides are normalised before
/// comparison. Verified byte-identical under PDFium builds 7690 and 7961.
fn normalise(s: &str) -> String {
    s.replace("\r\n", "\n")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extracts_text_from_a_real_pdf() {
    let _ =
        assert_extracts_the_sample_text("extracts_text_from_a_real_pdf", "sample.pdf", SAMPLE_PDF)
            .await;
}

/// The encrypted copies `pdf_pure_rust_fixture_extraction.rs` reads, through
/// PDFium, so that a PDFium-only build is held to them too. They hold the same
/// pages as `sample.pdf`, so they must extract to the same text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extracts_text_from_a_pdf_encrypted_with_an_empty_user_password() {
    for (name, bytes) in ENCRYPTED_EMPTY_USER_PASSWORD {
        let checked = assert_extracts_the_sample_text(
            "extracts_text_from_a_pdf_encrypted_with_an_empty_user_password",
            name,
            bytes,
        )
        .await;
        if !checked {
            // Skipped: libpdfium is unobtainable, and that is the cause for
            // every fixture, so one notice is enough.
            break;
        }
    }
}

/// `sample.pdf` encrypted by `mutool clean -E rc4-128` with `/Encrypt` left
/// inline in the trailer. The pure-rust backend cannot read it and its error
/// says "The pdf-pdfium backend reads such files"; this pins that claim.
const INLINE_ENCRYPT_PDF: &[u8] =
    include_bytes!("fixtures/pdf/sample.encrypted-inline-rc4-128.pdf");

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extracts_text_from_a_pdf_with_an_inline_encrypt_dictionary() {
    let _ = assert_extracts_the_sample_text(
        "extracts_text_from_a_pdf_with_an_inline_encrypt_dictionary",
        "sample.encrypted-inline-rc4-128.pdf",
        INLINE_ENCRYPT_PDF,
    )
    .await;
}

/// Zero pages is an error whichever backend reads the file, not only under
/// `pdf-pure-rust`: a PDF has at least one page, and storing such a file as an
/// empty document hides it from search while `add` reports success.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rejects_a_pdf_with_no_pages() {
    const TEST: &str = "rejects_a_pdf_with_no_pages";
    let pdf = pdf_of(
        &[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [] /Count 0 >>",
        ],
        "",
    );
    let result = PdfLoader.extract(&pdf, &document("no-pages.pdf")).await;
    match result {
        Err(e) if e.to_string().contains(LIBRARY_UNOBTAINABLE) => skip(TEST, &e),
        Err(LoaderError::ExtractionFailed(msg)) => {
            assert!(msg.contains("no pages"), "{msg}");
            assert!(!msg.contains("encrypt"), "{msg}");
        }
        other => panic!("zero pages is never a valid extraction: {other:?}"),
    }
}

/// The pure-rust backend's error for a file whose `/Encrypt` it cannot use
/// says "The pdf-pdfium backend reads such files" when the value is not an
/// encryption dictionary; this pins that for `/Encrypt null`. (pure-rust now
/// reads such a file itself, unless the key occurs more than once.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn extracts_text_from_a_pdf_whose_encrypt_entry_is_null() {
    const TEST: &str = "extracts_text_from_a_pdf_whose_encrypt_entry_is_null";
    match PdfLoader
        .extract(&encrypt_null_pdf(), &document("encrypt-null.pdf"))
        .await
    {
        Err(e) if e.to_string().contains(LIBRARY_UNOBTAINABLE) => skip(TEST, &e),
        Ok(LoaderOutput::Text(text)) => {
            assert!(text.contains(ENCRYPT_NULL_TEXT), "{text:?}");
        }
        other => panic!("an /Encrypt null file is not encrypted: {other:?}"),
    }
}

/// Fails in CI, where libpdfium must be obtainable, and otherwise prints the
/// skip notice. See the module docs.
fn skip(test: &str, e: &LoaderError) {
    assert!(
        !in_ci(),
        "{test}: libpdfium could not be obtained, so real-PDF extraction never ran — \
         and a CI lane must not report green on that. Provision the library and set \
         PDFIUM_LIB_PATH. Cause: {e}"
    );
    eprintln!(
        "SKIP {test}: libpdfium could not be obtained on this machine, so real-PDF \
         extraction was NOT verified. Set PDFIUM_LIB_PATH to an existing libpdfium \
         shared library to run it. Cause: {e}"
    );
}

/// Extracts `bytes` with [`PdfLoader`] and asserts the output is
/// `sample.expected.txt`. Returns `false`, after printing a notice naming
/// `test`, if libpdfium could not be obtained (see the module docs).
async fn assert_extracts_the_sample_text(test: &str, name: &str, bytes: &[u8]) -> bool {
    let output = match PdfLoader.extract(bytes, &document(name)).await {
        Ok(output) => output,
        Err(e) if e.to_string().contains(LIBRARY_UNOBTAINABLE) => {
            skip(test, &e);
            return false;
        }
        Err(e) => panic!("{name}: extraction of a valid PDF failed: {e}"),
    };

    let LoaderOutput::Text(text) = output else {
        panic!("{name}: the PDF loader must yield text, got {output:?}");
    };

    // `loaders::pdf` documents, as MEASURED, that PDFium separates intra-page
    // lines with CRLF. `normalise` below deliberately erases exactly that, so
    // without this assertion the divergence the module docs describe — and
    // that `sample.expected.pure-rust.txt` exists to contrast with — would be
    // unpinned, and a pdfium upgrade that switched to LF would slip through
    // green while the docs went stale.
    assert!(
        text.contains("\r\n"),
        "{name}: PDFium is documented to emit CRLF within a page; if that changed, \
         update the `loaders::pdf` module docs and both expected files. Got: {text:?}"
    );

    assert_eq!(
        normalise(&text),
        normalise(EXPECTED),
        "{name}: extracted text does not match fixtures/pdf/sample.expected.txt"
    );
    true
}
