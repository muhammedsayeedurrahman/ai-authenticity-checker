"""
Cyber crime complaint document generator.

Structured as a pre-filing worksheet that mirrors the actual filing
flow on the National Cyber Crime Reporting Portal (cybercrime.gov.in),
not a generic legal letter. The portal's real process (verified via web
search, Aug 2026 — see sources in the project chat log, not repeated
here): register with name/state/OTP-verified mobile number, then file a
4-part form — Incident Details, Suspect Details, Complainant Details,
Preview & Submit — and receive an acknowledgement number for tracking.
Complaints route through one of two tracks: "Report Cyber Crime related
to Women/Child" (sexual/objectionable content, NCII-style cases) or
"Report Other Cyber Crime" (fraud, impersonation, defamation, identity
theft). This document's sections mirror that structure section-for-
section so the fields transcribe directly onto the portal's own form.

Legal references (see LEGAL_REFERENCES below) were corrected against
current law as of Aug 2026:
  - India replaced the Indian Penal Code with the Bharatiya Nyaya
    Sanhita (BNS), 2023, effective 1 July 2024. The IPC sections
    commonly cited in older deepfake-complaint templates (419, 465,
    468, 500) are superseded — this module cites their current BNS
    equivalents (319, 336(3), 356) instead.
  - IT Act, 2000 Sections 66C/66D/66E were not touched by the BNS
    transition and remain current.
  - The Information Technology (Intermediary Guidelines and Digital
    Media Ethics Code) Amendment Rules, 2026 (Gazette notification
    G.S.R. 120(E), notified 10 Feb 2026, effective 20 Feb 2026)
    introduced "Synthetically Generated Information" as a defined,
    regulated category covering deepfakes/AI-generated media
    specifically — the first Indian rule to name this class of content
    directly, and the most relevant citation for an AI-detection tool's
    complaint output. Cited by notification, not by sub-rule number,
    since no source found during research quoted the exact sub-rule
    text with enough confidence to cite verbatim.

Presented as a starting reference throughout, not a certified legal
opinion — the document carries its own disclaimer to that effect, and
none of this is a substitute for the portal's own category selection
or a lawyer's judgment on a specific case.

This module only ever produces a *document* for the user to review and
submit themselves. It never transmits, files, or submits anything on
the user's behalf.

Output is HTML, not PDF: WeasyPrint (the PDF backend core/reports.py
already knows how to call) isn't installed, and its Windows setup
requires GTK system libraries that are painful to provision reliably.
The HTML is styled for both screen review and browser "Print to PDF" /
physical printing (@media print rules), so the missing PDF step is a
one-click action for the user rather than a blocker.
"""

from __future__ import annotations

import html
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

LEGAL_REFERENCES = [
    ("BNS, 2023 — Section 319", "Cheating by personation (impersonation) — supersedes IPC Section 419"),
    ("BNS, 2023 — Section 318(4)", "Cheating and dishonestly inducing delivery of property — supersedes IPC Section 420"),
    ("BNS, 2023 — Section 336(3)", "Forgery for the purpose of cheating — supersedes IPC Section 468"),
    ("BNS, 2023 — Section 356", "Defamation — supersedes IPC Sections 499/500"),
    ("IT Act, 2000 — Section 66C", "Identity theft (fraudulent use of a person's identity/likeness)"),
    ("IT Act, 2000 — Section 66D", "Cheating by personation using a computer resource"),
    ("IT Act, 2000 — Section 66E", "Violation of privacy (capturing/publishing private images)"),
    (
        "IT Rules (Amendment) 2026 — G.S.R. 120(E)",
        "Governs “Synthetically Generated Information” (deepfakes/AI-generated media) "
        "specifically — labelling, takedown, and intermediary obligations",
    ),
]

# Mirrors the portal's own top-level routing choice.
COMPLAINT_TRACKS = [
    (
        "Report Cyber Crime related to Women/Child",
        "Use this track if the content is sexual/obscene, or otherwise targets a woman or child "
        "(e.g. non-consensual intimate imagery, morphed/deepfake sexual content).",
    ),
    (
        "Report Other Cyber Crime",
        "Use this track for fraud, impersonation, identity theft, or defamation cases that don't "
        "fall under the Women/Child category above.",
    ),
]

FILING_STEPS = [
    "Go to cybercrime.gov.in and register with your name, state, and an OTP-verified mobile number.",
    "Choose the complaint track that matches Section 1 below (Women/Child or Other Cyber Crime).",
    "Fill the portal's 4-part form: Incident Details, Suspect Details, Complainant Details, then Preview & Submit — copy the corresponding fields from Sections 2-4 below.",
    "Upload a government ID proof (Aadhaar, PAN, or Passport — JPG/PNG, under 5MB) in Complainant Details. This is mandatory on the portal, not optional.",
    "Attach the original media file and, if useful as supporting evidence, this document, plus any screenshots/links showing where you encountered the content.",
    "Submit to receive an acknowledgement number. Save it — it's required (with OTP) to track your complaint's status afterward.",
]


def _esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")


def build_complaint_html(
    analysis: dict[str, Any],
    complainant: dict[str, Any],
    file_name: str = "",
    id_proof: Optional[dict[str, Any]] = None,
) -> str:
    """
    Build the complaint document as self-contained, printable HTML,
    structured to mirror the real cybercrime.gov.in filing flow — see
    module docstring.

    Args:
        analysis: the analysis result dict (same shape returned by
            core.pipeline.analyze_image / analyze_document / analyze_video —
            only a handful of common fields are read, so any of them work).
        complainant: {name, phone, email, address, incident_description}
        file_name: original media file name, for the record.
        id_proof: optional {data_uri, filename} for a government ID proof
            image (Aadhaar/PAN/Passport) the complainant uploaded — embedded
            in Section 2 as a visual record, mirroring the portal's own
            mandatory ID-proof-upload requirement in Complainant Details.
    """
    complaint_id = f"PXC-{uuid.uuid4().hex[:10].upper()}"

    verdict = analysis.get("verdict", "UNKNOWN")
    risk_percent = analysis.get("risk_percent", analysis.get("risk_score", 0) * 100)
    confidence = analysis.get("confidence", "")
    media_type = analysis.get("media_type", "image")
    analyzed_at = analysis.get("timestamp", _now_str())
    explanation = analysis.get("explanation", "")

    model_scores = analysis.get("model_scores", {}) or {}
    model_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{float(score) * 100:.1f}%</td></tr>"
        for name, score in model_scores.items()
    )
    if not model_rows:
        model_rows = "<tr><td colspan='2'>No per-model breakdown available for this analysis.</td></tr>"

    legal_rows = "".join(
        f"<tr><td>{_esc(section)}</td><td>{_esc(desc)}</td></tr>"
        for section, desc in LEGAL_REFERENCES
    )

    track_rows = "".join(
        f"<tr><td>{_esc(track)}</td><td>{_esc(desc)}</td></tr>"
        for track, desc in COMPLAINT_TRACKS
    )

    steps_html = "".join(f"<li>{_esc(step)}</li>" for step in FILING_STEPS)

    name = _esc(complainant.get("name", ""))
    phone = _esc(complainant.get("phone", ""))
    email = _esc(complainant.get("email", ""))
    address = _esc(complainant.get("address", "")).replace("\n", "<br>")
    incident_description = _esc(complainant.get("incident_description", "")) or (
        "[Describe where and how you encountered this content, and any "
        "harm or risk it poses, e.g. impersonation, fraud, harassment.]"
    )

    if id_proof and id_proof.get("data_uri"):
        id_proof_html = f"""
  <div class="id-proof-box">
    <div class="id-proof-label">Government ID Proof Attached — {_esc(id_proof.get('filename', ''))}</div>
    <img src="{id_proof['data_uri']}" alt="Government ID proof" class="id-proof-img">
  </div>"""
    else:
        id_proof_html = """
  <div class="id-proof-box id-proof-missing">
    <div class="id-proof-label">Government ID Proof — not attached</div>
    <p style="margin:4px 0 0;font-size:0.8rem;color:#8a6d00;">
      The portal requires a valid ID proof (Aadhaar, PAN, or Passport — JPG/PNG, under 5MB)
      to be uploaded in Complainant Details. Attach one when filing on the portal, or
      regenerate this worksheet with an ID proof image attached.
    </p>
  </div>"""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Cyber Crime Complaint Worksheet — {complaint_id}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    color: #1a1a1a;
    background: #f4f4f2;
    margin: 0;
    padding: 32px 16px;
    line-height: 1.55;
  }}
  .page {{
    max-width: 780px;
    margin: 0 auto;
    background: #ffffff;
    padding: 48px 56px;
    border: 1px solid #d8d8d4;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
  }}
  h1 {{ font-size: 1.35rem; text-align: center; margin: 0 0 4px; letter-spacing: 0.02em; }}
  .subtitle {{ text-align: center; color: #555; font-size: 0.85rem; margin-bottom: 4px; }}
  .meta {{ text-align: center; color: #777; font-size: 0.78rem; margin-bottom: 24px; }}
  h2 {{ font-size: 1.0rem; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 28px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.9rem; }}
  td {{ padding: 5px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  td:first-child {{ color: #444; width: 38%; font-weight: 600; }}
  .field-value {{ min-height: 1.2em; }}
  .verdict-badge {{
    display: inline-block; padding: 4px 12px; border-radius: 3px;
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.03em;
    background: #fdecea; color: #b91c1c; border: 1px solid #f5c2c0;
  }}
  .steps-box {{
    margin-top: 8px; padding: 14px 18px; background: #eef4ff;
    border: 1px solid #cfe0f7; border-radius: 4px;
  }}
  .steps-box ol {{ margin: 0; padding-left: 20px; font-size: 0.85rem; }}
  .steps-box li {{ margin: 4px 0; }}
  .id-proof-box {{
    margin-top: 10px; padding: 12px 14px; background: #f6faf6;
    border: 1px solid #cfe8cf; border-radius: 4px;
  }}
  .id-proof-box.id-proof-missing {{ background: #fffbea; border-color: #f3e2a0; }}
  .id-proof-label {{ font-size: 0.8rem; font-weight: 700; color: #2f6b2f; margin-bottom: 6px; }}
  .id-proof-missing .id-proof-label {{ color: #8a6d00; }}
  .id-proof-img {{ max-width: 320px; max-height: 220px; display: block; border: 1px solid #ccc; border-radius: 2px; }}
  .disclaimer {{
    margin-top: 32px; padding: 14px 16px; background: #fffbea;
    border: 1px solid #f3e2a0; border-radius: 4px; font-size: 0.8rem; color: #6b5a1e;
  }}
  .sign-row {{ display: flex; justify-content: space-between; margin-top: 56px; }}
  .sign-box {{ width: 45%; }}
  .sign-line {{ border-top: 1px solid #333; margin-top: 40px; padding-top: 4px; font-size: 0.8rem; color: #555; }}
  ul {{ font-size: 0.85rem; padding-left: 20px; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .page {{ box-shadow: none; border: none; max-width: none; padding: 24px 8px; }}
  }}
</style>
</head>
<body>
<div class="page">
  <h1>CYBER CRIME COMPLAINT — FILING WORKSHEET</h1>
  <div class="subtitle">Pre-filled for the National Cyber Crime Reporting Portal (cybercrime.gov.in) — transcribe into the portal's own form when filing</div>
  <div class="meta">Worksheet Reference: {complaint_id} &nbsp;|&nbsp; Prepared: {_now_str()}</div>

  <h2>How to File This on the Portal</h2>
  <div class="steps-box">
    <ol>{steps_html}</ol>
  </div>

  <h2>1. Which Complaint Track Applies</h2>
  <table>{track_rows}</table>

  <h2>2. Complainant Details <span style="font-weight:400;color:#888;">(portal: Complainant Details)</span></h2>
  <table>
    <tr><td>Full Name</td><td class="field-value">{name or '&nbsp;'}</td></tr>
    <tr><td>Mobile Number</td><td class="field-value">{phone or '&nbsp;'}<span style="color:#999;"> (OTP-verified on the portal at registration)</span></td></tr>
    <tr><td>Email</td><td class="field-value">{email or '&nbsp;'}</td></tr>
    <tr><td>Address</td><td class="field-value">{address or '&nbsp;'}</td></tr>
  </table>
  {id_proof_html}

  <h2>3. Incident Details <span style="font-weight:400;color:#888;">(portal: Incident Details)</span></h2>
  <p>Media flagged by automated forensic analysis as <strong>likely AI-generated or digitally manipulated</strong>:</p>
  <table>
    <tr><td>File Name</td><td>{_esc(file_name) or 'Not recorded'}</td></tr>
    <tr><td>Media Type</td><td>{_esc(media_type).title()}</td></tr>
    <tr><td>Verdict</td><td><span class="verdict-badge">{_esc(verdict)}</span></td></tr>
    <tr><td>Risk Score</td><td>{float(risk_percent):.1f}%</td></tr>
    <tr><td>Confidence</td><td>{_esc(confidence)}</td></tr>
    <tr><td>Analyzed At</td><td>{_esc(analyzed_at)}</td></tr>
  </table>
  <p style="margin-top:10px;">{incident_description}</p>

  <h2>4. Suspect Details <span style="font-weight:400;color:#888;">(portal: Suspect Details — fill in whatever is known)</span></h2>
  <table>
    <tr><td>Name / Handle</td><td class="field-value">&nbsp;</td></tr>
    <tr><td>Platform / URL</td><td class="field-value">&nbsp;</td></tr>
    <tr><td>Email / Phone (if known)</td><td class="field-value">&nbsp;</td></tr>
  </table>

  <h2>5. Technical Evidence <span style="font-weight:400;color:#888;">(attach with the complaint as supporting evidence)</span></h2>
  <p style="font-size:0.85rem;color:#444;">Produced by ProofyX automated forensic analysis. {_esc(explanation)}</p>
  <table>
    <tr><td colspan="2" style="font-weight:700;color:#222;">Per-Model Scores (P(AI-generated), 0-100%)</td></tr>
    {model_rows}
  </table>

  <h2>6. Applicable Legal Provisions (Reference Only)</h2>
  <table>{legal_rows}</table>
  <p style="font-size:0.78rem;color:#777;">India replaced the IPC with the Bharatiya Nyaya Sanhita (BNS), 2023 —
  the BNS sections above are current law, not the older IPC sections some templates still cite. These are
  commonly cited provisions in similar cases and are provided as a starting reference only — not a determination
  that they apply to this specific incident. Confirm with the portal's own category selection or a legal
  professional before filing.</p>

  <h2>7. Declaration</h2>
  <p>I declare that the information provided above is true and accurate to the best of my knowledge,
  and that the attached technical analysis was generated by ProofyX's automated detection system.
  I understand this analysis is a technical forensic aid and not a certified legal or expert
  opinion, and I am submitting this complaint at my own discretion.</p>

  <div class="sign-row">
    <div class="sign-box">
      <div class="sign-line">Signature</div>
    </div>
    <div class="sign-box">
      <div class="sign-line">Date</div>
    </div>
  </div>

  <div class="disclaimer">
    <strong>Disclaimer:</strong> This document was generated automatically by ProofyX based on an
    AI forensic analysis. It is a worksheet intended to help you file a complaint yourself on the
    National Cyber Crime Reporting Portal (cybercrime.gov.in) or with your local Cyber Crime Cell —
    ProofyX does not submit anything to any authority, and this document does not constitute legal
    advice or a certified forensic report. Review every field, complete the incident and suspect
    details, and verify current category/legal requirements against the portal itself or legal
    counsel before submission.
  </div>
</div>
</body>
</html>"""


def generate_complaint_document(
    analysis: dict[str, Any],
    complainant: dict[str, Any],
    file_name: str = "",
    id_proof: Optional[dict[str, Any]] = None,
) -> tuple[bytes, str, str]:
    """
    Returns (content_bytes, mime_type, suggested_filename).

    Tries a PDF render via WeasyPrint if installed; otherwise returns the
    print-ready HTML directly (see module docstring).
    """
    html_doc = build_complaint_html(analysis, complainant, file_name, id_proof)

    try:
        from weasyprint import HTML  # type: ignore
        pdf_bytes = HTML(string=html_doc).write_pdf()
        return pdf_bytes, "application/pdf", "cyber_complaint.pdf"
    except ImportError:
        return html_doc.encode("utf-8"), "text/html", "cyber_complaint.html"
