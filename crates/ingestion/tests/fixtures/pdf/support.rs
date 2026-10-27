//! Helpers shared by the PDF fixture tests, pulled in like `encrypted.rs`
//! with `#[path = "fixtures/pdf/support.rs"] mod support;`.

use cognee_models::{DataPoint, Document};
use uuid::Uuid;

/// The `Document` the PDF loader is handed for a file called `name`.
pub fn document(name: &str) -> Document {
    Document {
        base: DataPoint::new("PdfDocument", None),
        document_type: "pdf".to_string(),
        name: name.to_string(),
        raw_data_location: format!("file:///{name}"),
        mime_type: "application/pdf".to_string(),
        extension: "pdf".to_string(),
        data_id: Uuid::new_v4(),
        external_metadata: None,
    }
}

/// A classic-xref PDF whose objects are `objects`, numbered from 1, with
/// object 1 as the catalog and `trailer_extra` added to the trailer
/// dictionary.
pub fn pdf_of(objects: &[&str], trailer_extra: &str) -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
    }
    let xref = pdf.len();
    let size = objects.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {size}\n0000000000 65535 f \n").as_bytes());
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {size} /Root 1 0 R {trailer_extra}>>\nstartxref\n{xref}\n%%EOF\n"
        )
        .as_bytes(),
    );
    pdf
}

/// The text on the page of [`encrypt_null_pdf`].
pub const ENCRYPT_NULL_TEXT: &str = "Encrypt null is no encryption";

/// An unencrypted one-page PDF with `/Encrypt null` in its trailer. A `null`
/// value is the same as an absent entry (PDF 32000-1 §7.3.7), so every
/// backend must read it; `lopdf` alone does not, unless helped.
pub fn encrypt_null_pdf() -> Vec<u8> {
    let content = format!("BT /F1 12 Tf 72 720 Td ({ENCRYPT_NULL_TEXT}) Tj ET");
    pdf_of(
        &[
            "<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            &format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ],
        "/Encrypt null ",
    )
}
