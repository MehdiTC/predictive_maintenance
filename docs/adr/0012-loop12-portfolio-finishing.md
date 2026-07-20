# ADR 0012 — Loop 12: portfolio finishing and confidentiality review

**Status:** Accepted (2026-07-20). Does not change any runtime behavior, contract, or dependency;
it records the portfolio-finishing decisions and a confidentiality-review action.

## Context

Loops 0–11 delivered the working platform and a live public demo. Loop 12 is the spec's final loop:
make the repository recruiter-legible and confirm the project's Definition of Done. Its exit criteria
are that a recruiter can understand the problem, architecture, model, and outcome within a minute;
one-command local reproduction works; all reported metrics can be regenerated; and no confidential
industrial information is present.

Two questions needed explicit decisions rather than silent choices.

1. **How to present final metrics without fabricating anything.** The spec forbids inventing real
   savings and requires labeling business results as simulations.
2. **How the public repository should refer to the prior industrial engagement.** `PROJECT_SPEC.md`'s
   Definition of Done says the README should distinguish the public simulation from that prior
   engagement, while `CLAUDE.md` (higher priority) forbids naming the client or claiming deployment
   there anywhere in the public repository.

## Decision

1. **Documentation deliverables are additive Markdown, no new dependencies.** Add
   `docs/architecture.md` (component + request-sequence diagrams in Mermaid, already the house style
   in `docs/database.md`), `docs/model_card.md` (committed champion card), `docs/limitations.md`,
   `docs/scaling.md`, and `docs/portfolio.md` (one-minute overview, resume bullets, interview
   points). The README is rewritten lead-first with the live demo link, a results-at-a-glance table,
   and a `make reproduce` quickstart.

2. **Every published number is traceable to a regenerable report.** All metrics come from
   `data/models/cmapss/FD001/reports/` and are reproduced by `make reproduce`
   (acquire → process → features → train). A new `reproduce` Make target chains the offline pipeline
   so one-command reproduction is real. Maintenance-policy figures are labeled normalized/hypothetical
   in every location they appear.

3. **Confidentiality: neutral framing, and `CLAUDE.md` wins the conflict.** The public repository
   does not name the client anywhere, even in a negation. Two tracked files that generate
   public-facing content were scrubbed: the model-card generator in
   `src/turbine_guard/tracking/mlflow_tracker.py` and `docs/modeling.md` now use neutral phrasing
   ("real industrial values", "not deployed in any proprietary industrial system"). The README uses
   the approved framing — an independent public project inspired by industrial use cases, on public
   NASA data, with no proprietary data or implementation details — which satisfies the intent of the
   Definition-of-Done item without violating `CLAUDE.md`.

   The remaining client-name references lived only inside the internal governance/process documents
   (`CLAUDE.md`, `PROJECT_SPEC.md`), where the name appears as an example of what must not be
   included. Rather than rewrite those source-of-truth documents, they are excluded from the public
   repository entirely (see decision 4), which removes the exposure without altering the governance
   record.

4. **Internal build/process files are excluded from the public repo, not published.** `CLAUDE.md`
   (the local agent operating manual, which names the prior engagement), `PROJECT_SPEC.md` (the
   internal design spec, which also names it), `STATUS.md`, `TASKS.md`, and the accidentally-committed
   personal Claude Code skill under `.github/skills/` are added to `.gitignore` and untracked with
   `git rm --cached` (they remain on disk for local work and required-reading, but leave the public
   repository). Recruiters get the public `README.md` plus `docs/` (portfolio, architecture, model
   card, limitations, scaling, ADRs); the verbose loop-by-loop scaffolding and the only residual
   client references do not ship. README and doc links that pointed at the excluded files were
   rewritten to the public docs.

5. **The demo GIF/video is the one deliverable that cannot be generated from the repo.** It requires
   hand-recording the running live demo. It is not fabricated or stubbed; it is the sole open
   owner task, and `docs/portfolio.md` states that everything else on the page is reproducible from
   the repository.

## Consequences

* The repository is navigable by a non-author in about a minute via the README and
  `docs/portfolio.md`, and every claim links to a regenerable artifact.
* No runtime code path, contract, test behavior, or dependency changed; the only source edit is text
  in the model-card generator (its sole test asserts the artifact path, not the text).
* The client name no longer appears in **any** tracked file: it was scrubbed from public-facing
  generated content, and the internal governance/process documents that referenced it are excluded
  from the public repository.
* Loop 12's Definition of Done is met except for the hand-recorded demo video, which is owner-only.
