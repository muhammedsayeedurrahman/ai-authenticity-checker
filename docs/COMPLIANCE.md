# Compliance & Traceability Feature — India IT Rules 2026

Reference documentation for the compliance layer built across `core/compliance_label.py`,
`core/cybercrime_risk.py`, `core/sla.py`, `core/audit_hash.py`, `core/webhooks.py`,
`db/compliance_models.py`, `db/audit_log.py`, `db/compliance_repo.py`, `db/org_repo.py`,
`db/webhook_repo.py`, and `api/compliance_routes.py`.

**This document is engineering's reading of the regulation, not legal advice.**
Every label ProofyX produces carries this same disclaimer (see
`core/compliance_label.py::COMPLIANCE_DISCLAIMER`). Section 7, "Legal review
checklist," is intentionally unchecked — it must be signed off by counsel
before this feature is marketed as satisfying the amendment's obligations.

---

## 1. Regulatory basis

India's **Information Technology (Intermediary Guidelines and Digital Media
Ethics Code) Rules, 2026 amendment** (in force since **February 20, 2026**)
requires intermediaries operating in India to:

1. **Label** synthetically-generated information at the point of publication/upload.
2. Maintain **traceability** of that content back to its origin.
3. **Take down** flagged harmful deepfakes within **3 hours** of a valid complaint.

Sources: Mondaq (`IT Rules 2026 — deepfake regulation: three-hour takedowns and
AI labelling obligations`), Freshfields (`India targets deepfakes and AI-
generated content: key changes under MeitY's 2026`).

---

## 2. Label taxonomy → rule mapping

`core/compliance_label.py::build_compliance_label()` maps a pipeline
`risk_score`/`confidence` pair onto four codes, versioned as
`LABEL_RULESET_VERSION = "in-it-rules-2026.v1"`:

| `label_code` | Trigger | Maps to rule obligation |
|---|---|---|
| `synthetically_generated` | risk ≥ 0.60, confidence HIGH | Labeling (mandatory) + embedded metadata + traceability record |
| `possibly_synthetic` | 0.45 ≤ risk < 0.60, or risk ≥ 0.60 with LOW/MEDIUM confidence | Advisory label + routed to human review — the amendment does not clearly address borderline-confidence cases, so ProofyX does not auto-assert "synthetic" here |
| `no_synthetic_indicators` | risk < 0.45 | No labeling obligation triggered |
| `indeterminate` | analysis failed / no models loaded | Routed to manual review — never silently treated as clean |

**The 3-hour takedown SLA is deliberately narrower than "any synthetic
label."** `sla_applies` is only set when the content is `synthetically_generated`
**and** either (a) an existing `core/cybercrime_risk.py` fraud category fired
(`voice_clone_fraud`, `synthetic_identity`, `impersonation_video`) or (b) the
caller explicitly reports `flagged_by_complaint=True` at ingestion. This is
intentional: the takedown clock exists for *complained-about, harmful*
content, not every AI-generated image — ProofyX cannot infer that a
grievance was filed unless told.

**Why versioned:** every `content_labels` row stores `ruleset_version` and
`detector_version` alongside the determination. If the taxonomy above is
later corrected (e.g. after legal review), old records remain
reconstructable exactly as they were assessed — a v2 ruleset does not
retroactively invalidate v1 labels.

---

## 3. SLA clock semantics

`core/sla.py` — pure functions, no I/O:

- **Clock start = complaint-receipt time, not scan time.** `POST /compliance/content`'s
  `complaint_received_at` field takes precedence over "now" when the caller
  (the platform) knows when a grievance was actually filed. Getting this
  backwards would misrepresent the platform's actual compliance timeline to
  a regulator.
- **A backdated complaint can start an already-breached clock.** This is
  intentional and tested (`tests/test_sla.py::test_backdated_complaint_already_breached_at_ingestion`)
  — the honest outcome, not smoothed over.
- **Effective status is always derived at read time** (`clock_status()`) from
  `due_at` vs. now, not trusted from a possibly-stale stored `status` column.
  This means the background monitor (`core/sla_monitor.py`) being stopped or
  crashed can only delay a *notification* — `GET /compliance/sla` always
  reports the true status.
- **Resolution records truth, not flattery.** `SlaRepository.close_clock()`
  marks a clock `met` or `breached` based on whether the deadline had
  actually passed at the time of the platform's own action — never rounds a
  late action up to "met."
- **Resolution is a one-way transition, not idempotent-by-overwrite.**
  `close_clock()` raises `SlaClockAlreadyResolved` (surfaced as HTTP 409) if
  the clock isn't currently `running` — a duplicate or retried
  `POST /compliance/content/{id}/action` call can never silently flip an
  already-correct `met` record to `breached` (or vice versa) days later.
- **ProofyX never auto-takes-down content.** `POST /compliance/content/{id}/action`
  only *records* what the platform did (`removed`, `blocked`, `labeled`,
  `restored`, `cleared_false_positive`); the platform's own systems perform
  the actual takedown.

---

## 4. Audit trail — immutability boundary

`db/audit_log.py::AuditLog` exposes exactly three public methods —
`append`, `list`, `verify_chain` — no `update`, no `delete`, not even a
private one. Each entry hash-chains to the previous one
(`core/audit_hash.py`), so `verify_chain(org_id)` can detect the exact
sequence number of a row altered after the fact. The hashed material
covers `actor_type`, `actor_id`, and `subject_type` in addition to
`event_type`/`subject_id`/`payload` — attribution (*who did this*) is
tamper-evident too, not just the payload.

**Honest limitation:** this is enforced at the Python/ORM layer only. A
database administrator with direct SQL access could still `UPDATE` the
table — the hash chain makes tampering **detectable**, not **impossible**.
Hardening this to be actually tamper-*proof* (Postgres `BEFORE UPDATE OR
DELETE` triggers that raise, plus a restricted DB role with no UPDATE/DELETE
grant on this table) is deliberately deferred — call this out explicitly to
any customer whose compliance program requires the stronger guarantee.

**Concurrency:** appends are serialized per-org via an in-process
`asyncio.Lock` (`db/audit_log.py::_lock_for`). This is correct for a
single-instance deployment; running multiple API processes/workers needs a
DB-level lock (e.g. Postgres `SELECT ... FOR UPDATE` on a per-org cursor
row) instead, or two concurrent appends across processes could compute the
same `prev_hash` and corrupt the chain.

---

## 5. Data minimization (DPDP Act 2023)

`content_labels` stores `content_sha256`, **not the uploaded media**, by
default. Storing raw media plus an uploader identifier would make this row
a plausible target for a DPDP Act erasure request — which directly
conflicts with an append-only, hash-chained audit trail. `uploader_ref` is
an opaque, platform-supplied identifier (no PII enforced or expected), and
the `compliance_audit_log` table never stores personal data, only
system/content identifiers. Media retention beyond the hash is an explicit,
opt-in-per-org feature that does not exist yet — do not enable customer
media storage without a matching retention/erasure design.

---

## 6. Known MVP limitations (say these out loud to customers)

- **Multi-instance duplicate notifications.** `core/sla_monitor.py`'s poll
  loop and `core/webhooks.py::process_due_deliveries` both assume a single
  running instance. With >1 API worker, each runs its own loop and could
  double-notify a clock in a narrow race window. `SELECT ... FOR UPDATE
  SKIP LOCKED`-style claiming on Postgres is the multi-instance fix; not
  built yet.
- **Webhook secret encryption key is per-process when `PROOFYX_WEBHOOK_SECRET_KEY`
  isn't set.** `core/webhooks.py` falls back to an ephemeral in-process
  Fernet key with a logged warning. In any multi-worker deployment (the
  normal way FastAPI apps are actually run in production), each worker
  generates its own key — a secret encrypted by worker A becomes
  permanently undecryptable by worker B, silently breaking webhook signing
  for endpoints whose secret was created on a different worker than the one
  that later delivers to them. **Set `PROOFYX_WEBHOOK_SECRET_KEY` (a
  `Fernet.generate_key()` value) in any deployment with more than one
  worker process.**
- **API-key `scopes` and org-member `role` are stored but not enforced.**
  `CreateApiKeyRequest.scopes` and `OrgMember.role` (owner/admin/
  compliance_officer/viewer) exist in the schema, but every compliance
  route authorizes purely on org membership/key-ownership
  (`_require_org_match`) — a "viewer"-role member or a narrowly-scoped key
  currently has the same effective privileges as an owner/unscoped key
  within that org. Do not rely on either field as an access-control
  boundary until enforcement is added; today they're metadata only.
- **Webhook delivery is at-least-once, not exactly-once.** Consumers must
  be idempotent on `event_type` + the relevant subject id.
- **Audit immutability is detectable-tampering, not tamper-proof** (§4).
- **Generator/model attribution does not exist yet** — see §7. Nothing in
  this feature currently identifies *which* AI tool produced a fake, only
  that content is likely synthetic.
- **Detection accuracy is not perfect.** Recorded CorefakeNet fast-mode eval
  is ~82.5% accuracy / ~76% recall on a 332-sample held-out set — real
  fakes are missed at a non-trivial rate. This is why every label is framed
  as advisory input to the platform's own determination, never a
  certification, and why `possibly_synthetic` exists as a human-review
  tier instead of forcing a binary call.

---

## 7. Legal review checklist (UNCHECKED — do not claim compliance until signed off)

- [ ] Counsel has reviewed the label taxonomy in §2 against the amendment's
      actual text (not secondary summaries).
- [ ] Counsel has confirmed the 3-hour SLA's clock-start semantics (§3)
      match the amendment's own definition of "receipt of complaint."
  - [ ] Counsel has confirmed `content_sha256`-only storage (§5) satisfies
      the amendment's traceability requirement, or specified what
      additional retention is actually required.
- [ ] Counsel has reviewed the disclaimer language in
      `core/compliance_label.py::COMPLIANCE_DISCLAIMER` for adequacy.
- [ ] A data-protection review has covered `db/compliance_models.py`
      end-to-end for DPDP Act 2023 compliance, not just §5's summary.

---

## 8. Webhook integration reference (for customers)

Signature header: `X-Proofyx-Signature: t=<unix_ts>,v1=<hmac_sha256_hex>`,
computed over `f"{t}.{raw_request_body}"` using the endpoint's own secret
(shown once at registration — `POST /compliance/orgs/{org_id}/webhooks`).
Reference verifier: `core/webhooks.py::verify_signature()` — port this logic
directly; reject anything more than 5 minutes stale to prevent replay.

Event types currently emitted: `content.labeled`, `sla.started`,
`sla.due_soon`, `sla.breached`, `sla.resolved`. (`content.label_changed`,
`apikey.revoked`, and `audit.export_requested` are in the taxonomy but not
yet wired to a producer.)

Registration requires `https://` and a publicly-resolvable, non-private
address — see `core/webhooks.py::validate_webhook_url` for the exact SSRF
checks (blocks loopback, RFC1918 private ranges, link-local including the
`169.254.169.254` cloud-metadata address, multicast, reserved ranges, and
the RFC 6598 CGNAT/shared-address range `100.64.0.0/10`). At delivery time,
`deliver()` resolves the hostname once and connects directly to the
validated IP (rather than letting `httpx` re-resolve DNS independently)
to close a DNS-rebinding TOCTOU gap between validation and connection.

---

## 9. Generator / model attribution ("which tool made this fake")

Not part of the MVP. Split into two tracks per the original feature plan:

### 7a — Provenance-based (metadata), not yet wired

`core/metadata.py::check_c2pa()` is fully implemented (reads a C2PA
manifest via the optional `c2pa-python` package) and
`extract_full_metadata()` already calls it — **but only when given a
`file_path`**. Today, `core/pipeline.py`'s image-analysis functions call
`extract_full_metadata(image_pil)` with no `file_path`, because the
`/analyze/image` and `/compliance/content` upload paths decode the image
directly from an in-memory buffer rather than writing it to a temp file
first (unlike the video/audio paths, which already use
`tempfile.NamedTemporaryFile`).

**Remaining work, scoped but deliberately not done in this pass:** write
uploaded image bytes to a temp file before calling `extract_full_metadata`,
matching the existing video/audio pattern, and install+verify
`c2pa-python` in a real environment before relying on it — it is not
installed here and its API was not exercised against real signed content.
Wiring an unverified native binding into the image pipeline (which every
existing `/analyze/image` caller depends on) without being able to test it
end-to-end is the kind of speculative change this codebase's own
conventions warn against; it deserves its own dedicated, tested pass. This
gives high-precision, low-recall attribution (a named tool like "Adobe
Firefly" or "Midjourney" when metadata survives) — trivially defeated by
metadata stripping, so it complements rather than replaces the ML
ensemble.

### 7b — ML fingerprint attribution (research spike, go/no-go gated)

**Do not commit to this on a roadmap, pricing page, or customer contract
until the spike below passes.** Starting signals that already exist: the
7-model ensemble's per-model scores, CorefakeNet's 5 head scores plus
`attention_weights`, and the FFT-based frequency head (GAN/diffusion models
leave periodic spectral artifacts — the standard entry point for this kind
of research).

Why it's genuinely hard:
- **No dataset exists.** `dataset_portraits.py`'s training data is binary
  real/fake, not labeled by which generator produced each fake.
- **Generalization to unseen generators is an open research problem** —
  closed-set accuracy does not transfer, and new generators ship monthly,
  making this a permanent retraining obligation, not a one-time build.
- **CPU-only training reality** (per project memory: ~25 min/epoch for
  CorefakeNet) makes iteration slow.

**Proposed spike (~2 weeks) before any commitment:** label ~5k images
across 6 generators, train a multi-class head on frozen CorefakeNet
features, report **both** closed-set accuracy **and** leave-one-generator-
out (LOGO) accuracy. **Go/no-go gate: LOGO accuracy must clear a margin
large enough to be defensible in a compliance context** (a coin-flip-beating
number is not enough to put in front of a regulator). If it doesn't clear
that bar, ship only §7a and say so explicitly rather than overselling.
