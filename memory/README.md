# memory/

## Purpose

The project's **memory bank**: long-lived record of important design decisions, landed techniques, update log, and open-source references. Mandatory reading before any agent action (see `.cursor/rules/memory-first.mdc`).

## Files

- `code_design.md` — Architecture design and module decisions.
- `techniques.md` — Techniques that have landed in the codebase.
- `updates.md` — Chronological **important update log**.
- `research_first.md` — Open-source survey and alignment notes (references, deltas, rationale).

## Recording rules

- Record **important** items only: architectural decisions, added/removed modules, API changes, dependency upgrades, critical bugfixes, open-source references, selection conclusions.
- Do **not** record trivia: typo fixes, renames, ad-hoc debugging, reformatting.
- All entries use the same template:

  ```markdown
  ## YYYY-MM-DD - Title
  - Context:
  - Change:
  - Impact:
  - Follow-up:
  ```

- Append new entries at the **end** of the file in chronological order; never rewrite historical entries.
- Every commit or PR that carries an "important update" must update the relevant file here; otherwise it violates the `memory-first` rule.
