Audit and clean up Claude Code settings files.

Canonical usage: `/claude-cleanup [optional: specific settings file path]`

**Invoke the `claude-settings-cleanup` skill** via the Skill tool (`skill: "claude-settings-cleanup"`) to run the full review → report → act process.

The skill:

1. **Reviews** `.claude/settings.json` and `.claude/settings.local.json` (or the file path passed as an argument) for:
   - Secrets and credentials embedded in allow-list entries
   - High-risk wildcards allowing arbitrary code execution
   - Medium-risk destructive patterns
   - Stale, dead, and duplicate entries
   - Cross-project pollution
2. **Reports** findings by severity, concisely, with line numbers and recommended actions.
3. **Acts** only after the user picks a preset: everything / secrets + high-risk / secrets only / cherry-pick / report-only.

Use when the allow list has grown large, accumulated one-off approvals, contains credentials, or needs standardisation.
