Run a quick code review of recent or targeted changes.

Canonical usage: `/review [scope]`

For spec-driven feature audits, prefer `/review-task`.

Quick review flow:
1. identify the change scope
2. load relevant context from `CLAUDE.md` and `context/`
3. run the most relevant quality gates
4. report findings by severity with file references
