---
name: domain-finder
description: Check domain name availability across TLDs (.com, .io, .ai, .dev, .co, .app, .org, .net) and brainstorm creative domain ideas for a project or startup name. No API key needed — uses system whois and dig commands.
---

# /domain-finder

Check domain availability and brainstorm domain name ideas using system `whois` and `dig` commands. No API keys required.

## How to determine the mode

Look at `$ARGUMENTS`:

| Input looks like | Mode | Action |
|------------------|------|--------|
| A domain name (e.g. `loka.com`, `myapp.io`) | **Check** | Check that specific domain and its variants across TLDs |
| A project name or description (e.g. `task management app`) | **Brainstorm** | Generate creative domain ideas, then check availability |
| Empty or unclear | **Ask** | Ask the user what domain or project they want to explore |

---

## Mode: Check a specific domain

1. Extract the base name and TLD from the input.
2. Check availability across these TLDs: `.com`, `.io`, `.ai`, `.dev`, `.co`, `.app`, `.org`, `.net`
3. For each TLD, run this check using the Bash tool:

```bash
whois <domain>.<tld> 2>/dev/null | grep -iE "no match|not found|no data found|domain not found|available|No Object Found|NOT FOUND" | head -1
```

- If grep matches (exit 0 with output): domain is likely **available**
- If grep does NOT match (exit 1 / no output): domain is likely **taken** — extract registrar and expiry:

```bash
whois <domain>.<tld> 2>/dev/null | grep -iE "registrar:|expir|creation" | head -3
```

4. Also run a DNS check to confirm:

```bash
dig +short <domain>.<tld> A 2>/dev/null
```

- Empty result supports "available"; IP result supports "taken".

5. Present results in a clear table:

```
| Domain | Status | Details |
|--------|--------|---------|
| loka.com | Taken | Registrar: X, Expires: Y |
| loka.io | Available | - |
| loka.ai | Available | - |
```

**Run all whois/dig checks in parallel** (multiple Bash calls in one message) for speed.

---

## Mode: Brainstorm domain ideas

1. Take the project name or description from `$ARGUMENTS`.
2. Generate 15-20 creative domain name ideas across these categories:

| Category | Strategy | Examples for "task manager" |
|----------|----------|-----------------------------|
| **Direct** | Literal name + TLD | taskmanager.com, mytasks.io |
| **Brandable** | Short, memorable, invented | tasko.io, taskly.app |
| **Compound** | Two-word combos | taskflow.dev, cleartrack.io |
| **Metaphor** | Conceptual/abstract | launchpad.app, orbit.dev |
| **Prefix/Suffix** | get-, -hq, -app, try- | gettask.com, taskhq.io |

3. Check availability for **all generated names** using the same whois/dig method from Check mode.
4. Present results grouped by category, sorted with available domains first:

```
### Available
| Domain | Category |
|--------|----------|
| tasko.io | Brandable |
| cleartrack.dev | Compound |

### Taken
| Domain | Category | Registrar |
|--------|----------|-----------|
| taskflow.com | Compound | GoDaddy |
```

**Run all checks in parallel.**

---

## Important notes

- `whois` behavior varies by TLD registrar. Some TLDs (especially `.ai`, `.app`) may not return clean "not found" messages. When whois output is ambiguous, note it as "unclear — verify manually".
- Rate limiting: if checking many domains, batch in groups of 6-8 parallel calls to avoid throttling.
- Always remind the user: "whois checks are indicative, not authoritative. Verify on a registrar (Namecheap, Cloudflare, etc.) before purchasing."
- Keep the output concise. Lead with the results table, not the process.
