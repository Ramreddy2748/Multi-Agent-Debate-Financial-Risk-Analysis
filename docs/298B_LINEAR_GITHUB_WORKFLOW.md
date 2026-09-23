# Project 298B Linear + GitHub Workflow

This project uses Linear as the planning and ownership record, and GitHub as
the implementation, review, and merge record.

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
frontend
backend
data
ML/AI
agents
testing
documentation
evaluation
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
git checkout -b FR-12-fastapi-risk-api
git commit -m "FR-12 add FastAPI risk app backend"
git push origin FR-12-fastapi-risk-api
```

PR title:

```text
FR-12 Add FastAPI risk app backend
```

PR body should include:

- what changed
- why it changed
- how it was validated
- Linear issue link or `Closes FR-12`

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
