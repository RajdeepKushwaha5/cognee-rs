//! Pure-Rust PDF text extraction backend.
//!
//! Uses `pdf-extract` for text extraction without any native
//! dependencies. Works on all targets including Android, but may
//! produce lower-fidelity text for complex layouts, multi-column
//! documents, and some CJK/RTL encodings.
//!
//! Gated behind the `pdf-pure-rust` feature.

use super::format::format_document;
use super::startxref::repair_xref_offsets;
use crate::loaders::LoaderError;

/// Extract text from PDF bytes using the pure-Rust backend.
///
/// Returns the formatted text with page headers matching the Python
/// `pypdf_loader.py` output format. Unlike the pdfium backend, this
/// backend does not provide per-page error granularity -- if the
/// document-level parse fails, the entire extraction fails.
///
/// When the parse fails, xref offsets that point at whitespace just before
/// `xref` are corrected in a copy and the parse is retried once (see
/// [`super::startxref`]). A file that parses is never rewritten, and the
/// stored bytes are untouched either way.
///
/// A parse that "succeeds" with no pages at all is an error, as it is for
/// every backend (see [`format_document`]). `pdf-extract` stops at the first
/// page number it cannot find, so zero pages means the page tree, or page 1,
/// was unreadable. The known causes are encryption `lopdf` did not act on
/// (an inline `/Encrypt` dictionary) and a trailer `/Encrypt` that is not an
/// encryption dictionary at all, such as `/Encrypt null`. The second is
/// recovered from by ignoring the entry and parsing once more (see
/// [`recover_no_pages`]); the first is reported.
///
pub fn extract_text(bytes: &[u8]) -> Result<String, LoaderError> {
    // The bytes that were actually parsed: the input, or its repaired copy.
    let mut repaired = None;
    let page_texts = match pdf_extract::extract_text_from_mem_by_pages(bytes) {
        Ok(pages) => pages,
        Err(original) => {
            let Some(repair) = repair_xref_offsets(bytes) else {
                return Err(extraction_failed(original));
            };
            let corrections: Vec<String> = repair
                .corrections
                .iter()
                .map(|c| format!("{:?} {} -> {}", c.field, c.recorded, c.corrected))
                .collect();
            tracing::warn!(
                backend = "pdf-pure-rust",
                corrections = ?corrections,
                error = %original,
                "PDF xref offsets point at whitespace before the xref keyword; \
                 retrying with them moved onto the keyword"
            );
            // On a second failure the file's own error is the one to report:
            // the repaired copy is an internal attempt, not the document.
            let pages =
                pdf_extract::extract_text_from_mem_by_pages(&repair.bytes).map_err(|retry| {
                    tracing::debug!(error = %retry, "repaired PDF still failed to parse");
                    extraction_failed(original)
                })?;
            repaired = Some(repair.bytes);
            pages
        }
    };
    let parsed = repaired.as_deref().unwrap_or(bytes);

    // Zero pages: try the one known recovery, or find out why. Only reached
    // on this path, so a successful extraction is never parsed twice.
    let (page_texts, no_pages_detail) = if page_texts.is_empty() {
        match recover_no_pages(parsed) {
            Ok(pages) => (pages, String::new()),
            Err(detail) => (Vec::new(), detail),
        }
    } else {
        (page_texts, String::new())
    };

    let pages: Vec<(usize, Result<String, String>)> = page_texts
        .into_iter()
        .enumerate()
        .map(|(idx, text)| (idx + 1, Ok(text)))
        .collect();

    let text = format_document(&pages, &no_pages_detail)?;
    // `format_document` returns "" when every page was read but none had any
    // text, and nothing downstream treats that as a failure: ingest stores a
    // zero-byte document and reports success, the empty text embeds to a
    // zero-norm vector, and cosine KNN drops that row (see
    // `cognee_vector::zero_norm`). The whole pipeline then completes green
    // with nothing retrievable. That is worse than an error, so at minimum it
    // must not be silent.
    if text.trim().is_empty() {
        tracing::warn!(
            backend = "pdf-pure-rust",
            page_count = pages.len(),
            "PDF extraction produced no text — this backend returns empty for some \
             real PDFs (Type3 or otherwise undecodable font encodings) as well as for \
             scanned/image-only documents. The document will be stored empty and will \
             not be retrievable by search. Build with the `pdf-pdfium` feature to read \
             these files."
        );
    }
    Ok(text)
}

/// For a parse of `bytes` that produced no pages: the pages of a second parse
/// when the cause is one that can be safely ignored, or else the detail for
/// [`format_document`]'s error, naming the cause when it is encryption.
///
/// The cause that can be ignored is a trailer `/Encrypt` that is not an
/// encryption dictionary (`null`, a number, a dangling reference, ...). The
/// file is not encrypted (PDF 32000-1 §7.3.7: a `null` value is the same as an
/// absent entry), but `lopdf` takes its encrypted-load path for any
/// `/Encrypt` and, unable to authenticate, loads none of the document's
/// objects. With the entry renamed (see [`neutralise_encrypt`]), the file
/// parses normally. That happens at most once, and only when the value is
/// known not to be an encryption dictionary.
///
/// Whenever `lopdf` does not decrypt a file with an `/Encrypt`, it loads none
/// of its pages (at most the object an indirect `/Encrypt` points to; unit
/// tests pin this), so an undecryptable file always ends up here, with or
/// without object streams. That is why nothing on the success path checks
/// for it.
fn recover_no_pages(bytes: &[u8]) -> Result<Vec<String>, String> {
    const UNREADABLE: &str = " (its page tree or first page is unreadable).";
    let Ok(doc) = pdf_extract::Document::load_mem(bytes) else {
        return Err(UNREADABLE.to_string());
    };
    match trailer_encrypt(&doc) {
        TrailerEncrypt::Absent if doc.was_encrypted() => Err(": the document was decrypted, but \
             its page tree or first page is unreadable."
            .to_string()),
        TrailerEncrypt::Absent => Err(UNREADABLE.to_string()),
        // Verified for an inline dictionary with an empty user password
        // (`tests/pdf_fixture_extraction.rs`). With a user password PDFium,
        // which is given none, cannot open it either, hence the condition.
        TrailerEncrypt::Inline => Err(": the document is encrypted with an inline /Encrypt \
             dictionary, which the pdf-pure-rust backend cannot decrypt. The pdf-pdfium \
             backend reads such files when they have no user (open) password."
            .to_string()),
        TrailerEncrypt::Unknown => Err(": the trailer's /Encrypt refers to an object the \
             pdf-pure-rust backend could not read, so it cannot tell whether, or how, the \
             document is encrypted, and cannot open it."
            .to_string()),
        TrailerEncrypt::NotEncryption => {
            let Some(neutralised) = neutralise_encrypt(bytes) else {
                // Verified in `tests/pdf_fixture_extraction.rs` for a single
                // `/Encrypt null`.
                return Err(
                    ": the trailer's /Encrypt entry is not an encryption dictionary, \
                     and the pdf-pure-rust backend reads no objects from a file with one. \
                     The pdf-pdfium backend reads such files."
                        .to_string(),
                );
            };
            match pdf_extract::extract_text_from_mem_by_pages(&neutralised) {
                Ok(pages) if !pages.is_empty() => {
                    tracing::warn!(
                        backend = "pdf-pure-rust",
                        "PDF trailer has an /Encrypt entry that is not an encryption \
                         dictionary; read the document as unencrypted"
                    );
                    Ok(pages)
                }
                retry => {
                    tracing::debug!(?retry, "ignoring a non-dictionary /Encrypt did not help");
                    Err(
                        " (its page tree or first page is unreadable; its trailer's /Encrypt \
                         entry, which is not an encryption dictionary, was ignored)."
                            .to_string(),
                    )
                }
            }
        }
    }
}

/// What a loaded document's trailer says about encryption, as far as it
/// matters once a parse found no pages.
#[derive(Debug, PartialEq)]
enum TrailerEncrypt {
    /// No `/Encrypt`: the file was not encrypted, or `lopdf` decrypted it at
    /// load time and removed the entry.
    Absent,
    /// An inline encryption dictionary, which `lopdf` never decrypts.
    Inline,
    /// A value that cannot be an encryption dictionary: `null`, a number, a
    /// name, a string, an array, a reference to an object that is not in the
    /// xref, or to one `lopdf` read and found not to be a dictionary.
    NotEncryption,
    /// A reference to an object `lopdf` did not load, such as one it could
    /// not parse. It may be an encryption dictionary, so it is never ignored.
    ///
    /// A reference that does resolve to a dictionary also lands here, though
    /// it cannot reach this path: `pdf-extract` then tries to decrypt with an
    /// empty password and that parse fails outright (pinned by a unit test).
    Unknown,
}

fn trailer_encrypt(doc: &pdf_extract::Document) -> TrailerEncrypt {
    use pdf_extract::Object;
    match doc.trailer.get(b"Encrypt") {
        Err(_) => TrailerEncrypt::Absent,
        Ok(Object::Dictionary(_)) => TrailerEncrypt::Inline,
        Ok(Object::Reference(id)) => match doc.objects.get(id) {
            Some(Object::Dictionary(_)) => TrailerEncrypt::Unknown,
            Some(_) => TrailerEncrypt::NotEncryption,
            None if doc.reference_table.get(id.0).is_none() => TrailerEncrypt::NotEncryption,
            None => TrailerEncrypt::Unknown,
        },
        Ok(_) => TrailerEncrypt::NotEncryption,
    }
}

/// `bytes` with its `/Encrypt` key renamed to `/Encryp_` (same length, so
/// every xref offset stays valid), or `None` unless the key occurs exactly
/// once. With more than one, as after an incremental update, there is no
/// telling which trailer the value came from, so nothing is changed.
fn neutralise_encrypt(bytes: &[u8]) -> Option<Vec<u8>> {
    const KEY: &[u8] = b"/Encrypt";
    // A name ends at whitespace or a delimiter; `/EncryptMetadata` is another
    // key and must not match.
    let ends_name = |b: u8| b.is_ascii_whitespace() || b"()<>[]{}/%".contains(&b);
    let mut found = bytes
        .windows(KEY.len() + 1)
        .enumerate()
        .filter(|(_, w)| w.starts_with(KEY) && ends_name(w[KEY.len()]))
        .map(|(at, _)| at);
    let at = found.next()?;
    if found.next().is_some() {
        return None;
    }
    let mut out = bytes.to_vec();
    out[at + KEY.len() - 1] = b'_';
    Some(out)
}

fn extraction_failed(e: pdf_extract::OutputError) -> LoaderError {
    LoaderError::ExtractionFailed(format!("Failed to extract PDF text: {e}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "test code")]
mod tests {
    use super::super::format::format_pages;
    use super::*;

    /// A one-page PDF plus one incremental update per entry of `shorts`, each
    /// replacing the page's content stream, laid out like the SDK-1193 data
    /// sheets: classic xref tables, `/Prev`, and a blank line between the last
    /// `endobj` and each update's `xref`. Update `i` records its `xref` offset
    /// `shorts[i]` bytes short, in its `startxref` and in the next update's
    /// `/Prev`, as a writer that reuses its recorded value does.
    fn incremental_update_pdf(shorts: &[usize]) -> Vec<u8> {
        fn object(pdf: &mut Vec<u8>, offsets: &mut Vec<(u32, usize)>, id: u32, body: &str) {
            offsets.push((id, pdf.len()));
            pdf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
        }
        fn stream(text: &str) -> String {
            let content = format!("BT /F1 12 Tf 72 720 Td ({text}) Tj ET");
            format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            )
        }
        fn page(contents: u32) -> String {
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 4 0 R >> >> /Contents {contents} 0 R >>"
            )
        }
        fn entry(offset: usize) -> String {
            format!("{offset:010} 00000 n \n")
        }

        let mut pdf = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        object(
            &mut pdf,
            &mut offsets,
            1,
            "<< /Type /Catalog /Pages 2 0 R >>",
        );
        object(
            &mut pdf,
            &mut offsets,
            2,
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        );
        object(&mut pdf, &mut offsets, 3, &page(5));
        object(
            &mut pdf,
            &mut offsets,
            4,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        );
        object(&mut pdf, &mut offsets, 5, &stream("Original revision"));
        let first_xref = pdf.len();
        pdf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
        for (_, offset) in &offsets {
            pdf.extend_from_slice(entry(*offset).as_bytes());
        }
        pdf.extend_from_slice(
            format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{first_xref}\n%%EOF\n")
                .as_bytes(),
        );

        let mut prev = first_xref;
        for (i, &short) in shorts.iter().enumerate() {
            let contents = 6 + i as u32;
            let mut update = Vec::new();
            let text = format!("Revision {} SDK-1193", i + 1);
            object(&mut pdf, &mut update, contents, &stream(&text));
            object(&mut pdf, &mut update, 3, &page(contents));
            pdf.push(b'\n');
            let xref = pdf.len();
            pdf.extend_from_slice(b"xref\n0 1\n0000000000 65535 f \n");
            update.sort();
            for (id, offset) in &update {
                pdf.extend_from_slice(format!("{id} 1\n{}", entry(*offset)).as_bytes());
            }
            let recorded = xref - short;
            let size = contents + 1;
            pdf.extend_from_slice(
                format!(
                    "trailer\n<< /Size {size} /Root 1 0 R /Prev {prev} >>\n\
                     startxref\n{recorded}\n%%EOF\n"
                )
                .as_bytes(),
            );
            prev = recorded;
        }
        pdf
    }

    #[test]
    fn the_fixtures_reproduce_the_unpatched_failure() {
        // Guards the tests below: if `pdf-extract` starts accepting these files
        // on its own, they would pass without exercising the repair.
        for shorts in [&[1][..], &[1, 1], &[1, 0]] {
            let err = pdf_extract::extract_text_from_mem_by_pages(&incremental_update_pdf(shorts))
                .expect_err("pdf-extract 0.12 rejects an off-by-one xref offset");
            assert!(
                err.to_string().contains("invalid file trailer"),
                "{shorts:?}: {err}"
            );
        }
    }

    #[test]
    fn reads_an_incremental_update_whose_final_startxref_is_one_byte_short() {
        let text = extract_text(&incremental_update_pdf(&[1])).expect("repaired file parses");
        assert!(text.contains("Revision 1 SDK-1193"), "{text:?}");
        assert!(!text.contains("Original revision"), "{text:?}");
        assert_eq!(text, extract_text(&incremental_update_pdf(&[0])).unwrap());
    }

    #[test]
    fn reads_a_file_updated_twice_with_short_offsets() {
        // The second update's `/Prev` carries the first one's short offset.
        let text = extract_text(&incremental_update_pdf(&[1, 1])).expect("repaired file parses");
        assert!(text.contains("Revision 2 SDK-1193"), "{text:?}");
        assert!(!text.contains("Revision 1 SDK-1193"), "{text:?}");
        assert_eq!(
            text,
            extract_text(&incremental_update_pdf(&[0, 0])).unwrap()
        );

        let text = extract_text(&incremental_update_pdf(&[1, 0])).expect("short /Prev only");
        assert!(text.contains("Revision 2 SDK-1193"), "{text:?}");
    }

    #[test]
    fn well_formed_incremental_updates_extract_as_before() {
        for shorts in [&[0][..], &[0, 0]] {
            let pdf = incremental_update_pdf(shorts);
            let direct = pdf_extract::extract_text_from_mem_by_pages(&pdf).unwrap();
            let pages: Vec<_> = direct
                .into_iter()
                .enumerate()
                .map(|(i, t)| (i + 1, Ok(t)))
                .collect();
            assert_eq!(extract_text(&pdf).unwrap(), format_pages(&pages));
        }
    }

    /// A classic-xref PDF whose objects are `objects`, numbered from 1, with
    /// object 1 as the catalog.
    fn pdf_of(objects: &[&str]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let xref = pdf.len();
        pdf.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for offset in offsets {
            pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        pdf
    }

    /// One page with an empty content stream: what a scanned page looks like
    /// to a text extractor, minus the image.
    fn blank_page_pdf() -> Vec<u8> {
        pdf_of(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
            "<< /Length 0 >>\nstream\n\nendstream",
        ])
    }

    /// `sample.pdf` encrypted by `mutool clean -E rc4-128` and left as mutool
    /// wrote it, with `/Encrypt` inline in the trailer. `lopdf` 0.42 only
    /// decrypts an indirect `/Encrypt`; see `tests/fixtures/pdf/README.md`.
    const INLINE_ENCRYPT_PDF: &[u8] =
        include_bytes!("../../../tests/fixtures/pdf/sample.encrypted-inline-rc4-128.pdf");

    /// No pages at all: an empty page tree.
    fn no_pages_pdf() -> Vec<u8> {
        pdf_of(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [] /Count 0 >>",
        ])
    }

    /// [`ENCRYPTED_BLANK_PDF`] with its (cleartext) page tree emptied in place:
    /// `/Count 1/Kids[4 0 R]` becomes the same-length `/Count 0/Kids[     ]`,
    /// so every xref offset stays valid. It decrypts, then has no pages.
    fn encrypted_no_pages_pdf() -> Vec<u8> {
        const FROM: &[u8] = b"/Count 1/Kids[4 0 R]";
        const TO: &[u8] = b"/Count 0/Kids[     ]";
        let mut pdf = ENCRYPTED_BLANK_PDF.to_vec();
        let at = pdf
            .windows(FROM.len())
            .position(|w| w == FROM)
            .expect("the fixture's page tree is written as `/Count 1/Kids[4 0 R]`");
        pdf[at..at + FROM.len()].copy_from_slice(TO);
        pdf
    }

    /// A blank one-page PDF encrypted with an empty user password, with an
    /// indirect `/Encrypt` that `lopdf` does decrypt.
    const ENCRYPTED_BLANK_PDF: &[u8] =
        include_bytes!("../../../tests/fixtures/pdf/blank.encrypted-rc4-128.pdf");

    #[test]
    fn the_empty_extraction_fixtures_reproduce_what_pdf_extract_does() {
        // Guards the tests below: each input must reach `extract_text` as an
        // Ok with no text, either zero pages (which this module turns into an
        // error) or one empty page (which it keeps). If `pdf-extract` starts
        // failing (or reading) these on its own, the tests below would pass
        // without exercising anything.
        let pages = pdf_extract::extract_text_from_mem_by_pages(INLINE_ENCRYPT_PDF).unwrap();
        assert!(pages.is_empty(), "{pages:?}");

        for pdf in [no_pages_pdf(), encrypted_no_pages_pdf()] {
            let pages = pdf_extract::extract_text_from_mem_by_pages(&pdf).unwrap();
            assert!(pages.is_empty(), "{pages:?}");
        }

        for pdf in [ENCRYPTED_BLANK_PDF, &blank_page_pdf()] {
            let pages = pdf_extract::extract_text_from_mem_by_pages(pdf).unwrap();
            assert_eq!(pages.len(), 1);
            assert!(pages[0].trim().is_empty(), "{pages:?}");
        }
    }

    #[test]
    fn a_pdf_with_no_pages_is_an_error() {
        let err = extract_text(&no_pages_pdf()).expect_err("zero pages is never valid");
        assert!(matches!(err, LoaderError::ExtractionFailed(_)), "{err}");
        let msg = err.to_string();
        assert!(
            msg.contains("page tree or first page is unreadable"),
            "{msg}"
        );
        assert!(!msg.contains("encrypt"), "{msg}");
    }

    #[test]
    fn an_inline_encrypt_dictionary_is_an_error_not_an_empty_document() {
        let err = extract_text(INLINE_ENCRYPT_PDF).expect_err("unreadable, so it must fail");
        assert!(matches!(err, LoaderError::ExtractionFailed(_)), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("cannot decrypt"), "{msg}");
        assert!(msg.contains("pdf-pdfium"), "{msg}");
    }

    /// One page reading "Encrypt entry ignored", unencrypted, with
    /// `/Encrypt {entry}` added to the trailer. Object 4 is the content
    /// stream and object 5 the font.
    fn pdf_with_trailer_encrypt(entry: &str) -> Vec<u8> {
        let content = "BT /F1 12 Tf 72 720 Td (Encrypt entry ignored) Tj ET";
        pdf_with_trailer_encrypt_and(
            entry,
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            &format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ),
        )
    }

    /// [`pdf_with_trailer_encrypt`] with the given page tree (object 2) and
    /// object 4.
    fn pdf_with_trailer_encrypt_and(entry: &str, pages: &str, object_4: &str) -> Vec<u8> {
        let pdf = pdf_of(&[
            "<< /Type /Catalog /Pages 2 0 R >>",
            pages,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            object_4,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]);
        let pdf = String::from_utf8(pdf).unwrap();
        pdf.replacen("/Root 1 0 R", &format!("/Root 1 0 R /Encrypt {entry}"), 1)
            .into_bytes()
    }

    /// Values of a trailer `/Encrypt` that cannot be an encryption
    /// dictionary: `99 0 R` is not in the xref, and object 4 is a stream.
    const NOT_ENCRYPTION: [&str; 6] = ["null", "42", "/Junk", "(text)", "99 0 R", "4 0 R"];

    #[test]
    fn a_non_dictionary_encrypt_entry_makes_pdf_extract_read_nothing() {
        // Guards the test below: `lopdf` takes its encrypted-load path for
        // any trailer `/Encrypt`, loads none of the pages when it cannot
        // authenticate, and `pdf-extract` returns zero pages.
        for entry in NOT_ENCRYPTION {
            let pdf = pdf_with_trailer_encrypt(entry);
            let pages = pdf_extract::extract_text_from_mem_by_pages(&pdf);
            assert!(
                matches!(&pages, Ok(p) if p.is_empty()),
                "/Encrypt {entry}: {pages:?}"
            );
        }
    }

    #[test]
    fn a_non_dictionary_encrypt_entry_is_ignored_and_the_document_extracted() {
        // Such a file is not encrypted (a `null` entry is the same as none),
        // so it is read as it would be without the entry.
        for entry in NOT_ENCRYPTION {
            let text = extract_text(&pdf_with_trailer_encrypt(entry))
                .unwrap_or_else(|e| panic!("/Encrypt {entry}: {e}"));
            assert!(
                text.starts_with("Page 1:") && text.contains("Encrypt entry ignored"),
                "/Encrypt {entry}: {text:?}"
            );
        }
    }

    #[test]
    fn a_non_dictionary_encrypt_entry_is_not_blamed_when_the_page_tree_is_broken() {
        let pdf = pdf_with_trailer_encrypt_and(
            "null",
            "<< /Type /Pages /Kids [] /Count 0 >>",
            "<< /Length 0 >>\nstream\n\nendstream",
        );
        let err = extract_text(&pdf).expect_err("still no pages");
        assert!(matches!(err, LoaderError::ExtractionFailed(_)), "{err}");
        let msg = err.to_string();
        assert!(
            msg.contains("page tree or first page is unreadable"),
            "{msg}"
        );
        assert!(msg.contains("was ignored"), "{msg}");
        assert!(!msg.contains("cannot decrypt"), "{msg}");
    }

    #[test]
    fn an_encrypt_entry_is_only_neutralised_when_it_occurs_once() {
        let once = pdf_with_trailer_encrypt("null");
        let neutralised = neutralise_encrypt(&once).expect("one /Encrypt");
        assert_eq!(neutralised.len(), once.len());
        let renamed = String::from_utf8(neutralised).unwrap();
        assert!(renamed.contains("/Encryp_ null"), "{renamed}");
        assert!(!renamed.contains("/Encrypt"), "{renamed}");

        // Twice, as after an incremental update: no telling which one the
        // parser used, so nothing is changed and the file is reported.
        let twice =
            String::from_utf8(once)
                .unwrap()
                .replacen("%PDF-1.7\n", "%PDF-1.7\n% /Encrypt\n", 1);
        assert_eq!(neutralise_encrypt(twice.as_bytes()), None);
        // `/EncryptMetadata` is a different key.
        assert_eq!(neutralise_encrypt(b"<< /EncryptMetadata false >>"), None);
    }

    #[test]
    fn an_encrypt_reference_to_an_unreadable_object_is_never_ignored() {
        // Object 4 is in the xref but does not parse, so `lopdf` cannot tell
        // whether it was an encryption dictionary. Ignoring the entry could
        // then read ciphertext as text, so the file is reported instead.
        let pdf = pdf_with_trailer_encrypt_and(
            "4 0 R",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Filter",
        );
        let doc = pdf_extract::Document::load_mem(&pdf).unwrap();
        assert_eq!(trailer_encrypt(&doc), TrailerEncrypt::Unknown);
        let err = extract_text(&pdf).expect_err("unknown encryption");
        let msg = err.to_string();
        assert!(msg.contains("could not read"), "{msg}");
        assert!(!msg.contains("pdf-pdfium"), "{msg}");
    }

    /// `sample.pdf` encrypted with the user password `secret`, with an
    /// indirect `/Encrypt`.
    const USER_PASSWORD_PDF: &[u8] =
        include_bytes!("../../../tests/fixtures/pdf/sample.encrypted-user-password-rc4-128.pdf");

    #[test]
    fn an_encryption_dictionary_lopdf_keeps_never_reaches_the_no_pages_path() {
        // Why `TrailerEncrypt::Unknown` covers a reference that resolves to
        // a dictionary without a message of its own: `pdf-extract` sees
        // `is_encrypted()`, tries the empty password, and fails the parse.
        let err = pdf_extract::extract_text_from_mem_by_pages(USER_PASSWORD_PDF)
            .expect_err("the empty password does not open it");
        assert!(err.to_string().contains("password"), "{err}");
    }

    #[test]
    fn lopdf_loads_none_of_the_pages_of_a_file_it_does_not_decrypt() {
        // Why the undecryptable-encryption check only needs to run when there
        // are no pages: when the trailer has an `/Encrypt` that `lopdf` does
        // not act on, it loads none of the document's pages, so the page tree
        // is never visible as cleartext with the content streams left as
        // ciphertext. This holds whether or not the file uses object streams
        // (the inline fixture has none: its catalog and page tree are plain
        // dictionaries). If a `lopdf` upgrade starts loading such files
        // without decrypting them, this fails, and `extract_text` has to
        // check `/Encrypt` on every parse instead.

        // Inline: nothing at all is loaded.
        let doc = pdf_extract::Document::load_mem(INLINE_ENCRYPT_PDF).unwrap();
        assert!(doc.trailer.get(b"Encrypt").is_ok());
        assert!(doc.objects.is_empty(), "{:?}", doc.objects.keys());
        assert!(doc.get_pages().is_empty());

        // Indirect, with a password `lopdf` does not have: only the
        // encryption dictionary itself is loaded.
        let doc = pdf_extract::Document::load_mem(USER_PASSWORD_PDF).unwrap();
        let id = doc
            .trailer
            .get(b"Encrypt")
            .and_then(pdf_extract::Object::as_reference)
            .expect("an indirect /Encrypt");
        assert_eq!(doc.objects.keys().collect::<Vec<_>>(), vec![&id]);
        assert!(doc.get_dictionary(id).unwrap().get(b"Filter").is_ok());
        assert!(doc.get_pages().is_empty());
    }

    #[test]
    fn a_decrypted_pdf_with_no_pages_is_an_error_that_does_not_blame_decryption() {
        let err = extract_text(&encrypted_no_pages_pdf()).expect_err("zero pages");
        assert!(matches!(err, LoaderError::ExtractionFailed(_)), "{err}");
        let msg = err.to_string();
        assert!(msg.contains("was decrypted"), "{msg}");
        assert!(!msg.contains("cannot decrypt"), "{msg}");
    }

    #[test]
    fn an_encrypted_pdf_with_blank_pages_still_extracts_to_empty_text() {
        // As PDFium and pypdf do: encryption alone is no reason to reject a
        // document whose pages decrypt fine but carry no text.
        assert_eq!(extract_text(ENCRYPTED_BLANK_PDF).unwrap(), "");
    }

    #[test]
    fn an_unencrypted_pdf_with_blank_pages_still_extracts_to_empty_text() {
        // Image-only and scanned documents land here. Failing them would
        // reject files Python's pypdf loader accepts; they keep the warning.
        assert_eq!(extract_text(&blank_page_pdf()).unwrap(), "");
    }

    #[test]
    fn a_tail_the_repair_does_not_recognise_still_fails_cleanly() {
        // Six bytes short lands inside `endobj`: not whitespace, so the repair
        // declines and the parser's own error is reported.
        let err = extract_text(&incremental_update_pdf(&[6])).expect_err("still unreadable");
        assert!(matches!(err, LoaderError::ExtractionFailed(_)), "{err}");
        assert!(err.to_string().contains("invalid file trailer"), "{err}");
    }
}
