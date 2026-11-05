---
name: claude-settings-cleanup
description: Audit and clean Claude Code settings files — detects secrets, dangerous wildcards, stale entries, duplicates, and cross-project pollution in .claude/settings.json and settings.local.json. Presents findings with recommendations, then applies changes only after user direction. Use when the allow list has grown large, accumulated one-off approvals, contains credentials, or needs standardisation.
---

# claude-settings-cleanup

Reviews, standardises, and cleans up Claude Code settings files. The `permissions.allow` list is the main target — it grows over time as one-off approvals accumulate and can accidentally collect secrets, dangerous wildcards, and stale entries.

## Scope

Default targets (in order):
1. `.claude/settings.json` — project-level, git-tracked
2. `.claude/settings.local.json` — user-local, usually gitignored
3. Any other `settings*.json` under `.claude/`

If the user passes a path, check only that file.

**Never touch** without explicit instruction:
- `hooks` section
- `env` section
- `additionalDirectories` section
- `permissions.deny` (only add to it, never remove)
- MCP server entries
- `autoMemoryDirectory` and other top-level config keys

## Process

Three phases: **Review → Report → Act**. Do not apply changes until the user confirms.

---

### Phase 1 — Review

Read each target file. For every entry in `permissions.allow`, classify it into the buckets below. Keep track of line numbers so the report can cite them.

#### 1. Secrets (critical)

Scan every entry for credential-shaped substrings. Match patterns (case-insensitive):

- **API keys / tokens**: `sk-`, `Bearer `, `ghp_`, `gho_`, `xoxb-`, `AIzaSy`, `AKIA`, `ya29.`, `npm_`
- **JWT**: `eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.` — decode the middle segment with `base64 -d` and check the `exp` field
- **Connection strings with inline password**: `postgres://`, `postgresql://`, `postgresql+asyncpg://`, `mongodb://`, `mysql://`, `redis://` containing `:<value>@`
- **Raw secrets by flag**: `PGPASSWORD=`, `--deployment-token`, `--client-secret`, `--api-key`, `JWT_SECRET_KEY=`, `ANTHROPIC_API_KEY=`, `OPENAI_API_KEY=`, `AZURE_OPENAI_API_KEY=`, `AZURE_STORAGE_CONNECTION_STRING=`, `AccountKey=`, `SharedAccessKey=`
- **Generic high-entropy strings** (≥32 hex/base64 chars) adjacent to a secret-shaped flag

For every match, also check:
- Whether the file is git-ignored: `git check-ignore -v <file>`
- Whether it was ever committed: `git log --oneline --all -- <file>`
- If JWT, decode `exp` to see if it's expired

#### 2. High-risk wildcards (arbitrary code execution)

| Pattern | Why it's dangerous |
|---------|-------------------|
| `Bash(find:*)` | `find / -delete`, `find ... -exec rm {} \;` |
| `Bash(mv:*)` | Can move any file anywhere |
| `Bash(rm:*)` | Never allow |
| `Bash(cp:*)` | Can overwrite arbitrary files |
| `Bash(perl -pi -e:*)`, `Bash(sed -i:*)` | In-place edits to any file |
| `Bash(node:*)`, `Bash(node -e:*)` | Arbitrary JS execution |
| `Bash(python -c:*)`, `Bash(python3 -c:*)` | Arbitrary Python execution |
| `Bash(npx:*)` (unqualified) | Downloads and runs arbitrary npm packages |
| `Bash(kill:*)`, `Bash(pkill:*)` (unqualified) | Kill any user process |
| `Bash(xargs:*)` (unqualified) | Runs arbitrary commands from stdin |
| `Bash(eval:*)`, `Bash(exec:*)`, `Bash(sh -c:*)`, `Bash(bash -c:*)` | Escapes the permission system |
| `Bash(chmod:*)`, `Bash(chown:*)` | Privilege / ownership changes |
| `Bash(curl:*)` (unqualified host) | Can POST/DELETE anywhere, including prod |

#### 3. Medium-risk wildcards

Flag but do not auto-remove:

- `Bash(git checkout:*)` — `git checkout -- .` discards uncommitted work
- `Bash(git update-index:*)` — can silently hide changes via `--skip-worktree`
- `Bash(git reset:*)`, `Bash(git clean:*)` — destructive
- `Bash(psql ... -c:*)` — arbitrary SQL on whatever DB the connection string targets
- `Bash(pytest:*)`, `Bash(npm run:*)` — run in-repo code (usually fine, worth noting)
- `curl` to production URLs — prefix matching still allows any HTTP method

#### 4. Stale, dead, and duplicate entries

- **Expired JWTs** (from phase 1)
- **Debug artifacts**: `echo "EXIT=$?"`, `echo "---exit=$?---"`, `BUILD_EXIT=$?`, `TSC_EXIT:` — noise from past hook debugging
- **Shell loop fragments**: `Bash(done)`, `Bash(do echo:*)`, bare `Bash(then)`, `Bash(fi)` — leftover parsing of compound commands
- **One-off specific commands already covered** by a wildcard elsewhere in the file (e.g. `Bash(grep -n "foo" /specific/path)` when `Bash(grep:*)` exists)
- **Exact duplicates**
- **Cross-file duplicates** — `settings.local.json` repeating something already in `settings.json`

#### 5. Cross-project pollution

- Paths under directories other than the current project root (compare against `pwd`)
- References to unrelated repositories

---

### Phase 2 — Report

Produce a concise, scannable report organised by severity. Format:

```
## Review: .claude/settings.json (N entries)

### Critical — secrets (X findings)
- [line 188] `az deployment group create … postgresAdminPassword='***' jwtSecretKey='***'`
  - Gitignored: yes — Ever committed: no
  - Action: remove. No rotation needed (never left the laptop).

### High risk — arbitrary code execution (Y findings)
- [line 12] `Bash(find:*)` — allows `find / -delete`
  - Action: remove. Glob tool covers read-only use cases.
- [line 44] `Bash(node:*)` — arbitrary JS execution
  - Action: remove.

### Medium risk (Z findings)
- [line 32] `Bash(git checkout:*)` — `git checkout -- .` discards uncommitted work
  - Action: keep, or add `Bash(git checkout -- *)` to permissions.deny

### Noise — stale / duplicate / dead (W findings)
- [lines 120, 121, 124, 127] 4× `echo "EXIT=$?"` variants → collapse to `Bash(echo:*)`
- [line 10] `Bash(done)` — orphan shell-loop fragment, remove

### Cross-project pollution (V findings)
- [line 55] `/Users/.../OtherProject/...` — belongs to a different repo

## Summary
Total allow entries: N → proposed: M
- Secrets to remove: X
- High-risk wildcards to remove or narrow: Y
- Noise to collapse: W
- Medium-risk flagged for review: Z
```

After the report, ask **one** question — what to apply. Offer presets:

1. **Apply everything** — secrets + high-risk + noise + cross-project
2. **Secrets + high-risk only** — conservative safety fix
3. **Secrets only** — minimum safety fix
4. **Cherry-pick** — walk through each section
5. **Report only** — no changes

Keep the report under ~50 lines if possible. If there are many findings, show the top 5 per category and add a `... and N more` footer.

---

### Phase 3 — Act

Based on the user's choice:

1. Re-read the target file (it may have changed since phase 1)
2. For large cleanups, use `Write` to rewrite the file in full
3. For surgical fixes, use `Edit`
4. Preserve `hooks`, `env`, `additionalDirectories`, `permissions.deny`, `autoMemoryDirectory`, and MCP entries exactly
5. After writing, report:
   - Entries before → after
   - What was removed, grouped by category
   - Any rotations the user should do (only if secrets were ever committed to git)
   - Any medium-risk items still in place that they may want to revisit

---

## Standardisation rules

When collapsing duplicates into wildcards, apply these conventions:

- **One wildcard per tool family**: prefer `Bash(grep:*)` over many specific greps
- **Narrow over broad**: `Bash(pkill -f uvicorn:*)` over `Bash(pkill:*)`
- **Scope curl to specific hosts**: `Bash(curl -s http://localhost:8000/*)`, not `Bash(curl:*)`
- **Explicit git subcommands**: `Bash(git status:*)`, `Bash(git diff:*)`, etc. — never `Bash(git:*)` unless the user asks
- **Group entries semantically** in the output file: read / search / process / git / package-managers / DB / HTTP / cloud-CLI

## Guardrails

- Never auto-remove entries from `permissions.deny`
- Never modify `hooks`, `env`, `additionalDirectories`, or `autoMemoryDirectory` unless asked
- Before flagging something as a secret, verify it's not a placeholder (`<redacted>`, `XXX`, `your-key-here`) or a documented public value
- If the target file is small (<20 entries) or already clean, say so and exit without a full report
- Secrets not in git history → recommend removal only, not rotation
- Secrets found in git history → recommend immediate rotation **and** `git filter-repo` / BFG guidance

## Output tone

Concise. Lead with the count and severity. No filler. The user should be able to decide what to do in under 30 seconds of reading.
