# Project 298B Linear + GitHub Workflow

This project uses Linear as the planning and ownership record, and GitHub as
the implementation, review, and merge record.

## Current Project Records

- Linear workspace: `298b-financial-risk-multi-agent`
- Linear team: `298B - Financial Risk Multi-Agent Project`
- Linear issue prefix: `298-*`
- Course workflow issue: `298-5 Finalize Linear and GitHub course workflow`
- GitHub repository: `Ramreddy2748/Data298A--Masters-Project`
- Local repository path: `/Users/varunreddyseelam/Documents/Data298A--Masters-Project`

## Required Setup

1. Create one shared Linear workspace for the team.
2. Invite every teammate using their actual email address.
3. Invite the Project 298B TA as Admin using the exact email from Canvas.
4. Create a Linear project named:

```text
Project 298B - Financial Risk Multi-Agent
```

5. Add this summary:

```text
Build a production-style multi-agent financial risk modeling system that ingests market, fundamental, macro, and news data; generates auditable stock risk verdicts; supports debate-based contradiction review; and exposes results through an API-backed frontend.
```

6. Configure statuses:

```text
Backlog
Todo
In Progress
In Review
Ready to Merge
Done
```

7. Add labels:

```text
agents
backend
data
documentation
evaluation
frontend
ML/AI
project-management
testing
deployment
```

8. Connect the GitHub repository to Linear.
9. Protect the default GitHub branch:

- require pull requests before merge
- require at least one independent approval
- require passing checks when checks are configured
- block direct pushes to the default branch

## Per-Task Workflow

Every meaningful contribution should follow this path:

```text
Linear issue -> In Progress -> branch -> commits -> PR -> review -> merge -> Done
```

Use the exact Linear issue ID in the branch, commit, and PR.

Example:

```bash
git checkout -b 298-5-finalize-linear-github-workflow
git commit -m "298-5 finalize Linear GitHub workflow docs"
git push origin 298-5-finalize-linear-github-workflow
```

PR title:

```text
298-5 Finalize Linear GitHub workflow docs
```

PR body should include:

- what changed
- why it changed
- how it was validated
- Linear issue link or `Closes 298-5`

## Current 298-5 Scope

Issue `298-5` is complete when the repository contains the workflow evidence
needed for Project 298B:

- this workflow document
- the Linear backlog seed CSV at `docs/linear_backlog_seed.csv`
- the GitHub pull request template at `.github/pull_request_template.md`
- `.gitignore` entries for local-only generated artifacts such as `.venv/`,
  `mlruns/`, `mlflow.db`, and `data/gold/_mlflow_artifacts/`

Validation for `298-5`:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
rows = list(csv.DictReader(Path("docs/linear_backlog_seed.csv").open()))
assert rows
assert {"Title", "Description", "Status", "Labels", "Priority"} <= set(rows[0])
print(f"Linear backlog CSV valid: {len(rows)} issues")
PY
```

## Issue Template

```markdown
## Objective

What concrete outcome should this issue produce?

## Context

Why is this needed for the project?

## Scope

What files, modules, screens, or pipeline steps are included?

## Acceptance Criteria

- [ ] Criteria 1
- [ ] Criteria 2
- [ ] Criteria 3

## Dependencies / Blockers

List any upstream work, credentials, data, or review dependencies.

## Required Evidence

- GitHub PR link
- test or command output
- screenshot, generated artifact, model metric, or report link when applicable
```

## Contribution Evidence Rules

- Do not use commit count as contribution evidence.
- Do not backfill all issues at the end of the semester.
- Do not self-approve PRs.
- Keep Linear status current while work is happening.
- Each student should own clear issues with testable acceptance criteria.

## 298B PR Checklist

Before merging a Project 298B PR:

- [ ] The branch name contains the Linear issue ID, for example `298-5-*`.
- [ ] The PR title starts with the Linear issue ID.
- [ ] The PR body includes `Closes 298-5` or the relevant issue ID.
- [ ] The PR lists validation commands or other evidence.
- [ ] Another teammate reviews and approves the PR.
- [ ] The Linear issue moves to `Done` only after acceptance criteria are met.
