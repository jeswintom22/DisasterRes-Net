# Hierarchy Invariants

The 5 rules the validator enforces, what they mean, where they're stored,
and where in code they can be violated. Keep this file in sync with
[scripts/validate.py](scripts/validate.py).

The Loka data model is built on 5 primitives: Entity, Role, Task, Event,
Artifact. The hierarchy chain is:

```
organisation → project → mission → activity → time_entry
```

Every link in the chain is one-to-many, walking downward — a project belongs
to **exactly one** organisation, a mission belongs to **exactly one**
project, and so on. The chain must never be short-circuited; e.g. a time
entry must reach a mission via an activity, never directly.

---

## Rule 1 — project belongs to exactly 1 organisation

**Statement**: every active row in `entity_projects` has exactly one active
`belongs_to` relationship targeting an `entity_organisations` row.

**Why**: a project is the unit a customer pays for / a department owns.
Without an org link, billing, access control, and reporting are all broken.

**Where it's stored**: `relationships` table.

```
rel_type     = 'belongs_to'
source_id    = entity_projects.id   (the project)
target_id    = entity_organisations.id   (the organisation)
valid_to     = NULL  (active)
```

Historically two writer shapes have existed in prod and both are accepted
by the validator:

- canonical: `source_type='project', target_type='organisation'`
- legacy:    `source_type='entity',  target_type='entity'`

New writes should use the canonical shape.

**Where it can be violated in code**:

- [app/backend/app/api/entities.py](../../../app/backend/app/api/entities.py) — `create_project_with_groups()`. This is the canonical project-create path. Must validate that `organisation_id` is provided and points to an org in the same tenant, then create the `belongs_to` edge in the same transaction.
- [app/backend/app/api/bulk_import.py](../../../app/backend/app/api/bulk_import.py) — bulk project import. Same constraint must hold.
- [app/backend/scripts/](../../../app/backend/scripts/) — any seed/import script that inserts projects. They must write the `belongs_to` edge; otherwise the project lands orphaned.

**How the validator queries it**: see `rule1_project_to_org()` in
[scripts/validate.py](scripts/validate.py).

---

## Rule 2 — mission belongs to exactly 1 project

**Statement**: every row in `entity_missions` has exactly one active
`has_mission` relationship from a project (the project is `source_id`,
the mission is `target_id`). This applies to both top-level missions and
subtasks (subtasks also have a `parent_id` to their parent mission, but
that does not satisfy Rule 2 — every mission still needs its own
`has_mission` link to the project).

**Why**: a mission represents work being done for someone. Without a
project, there's nothing to bill against, no team to assign to, and no
way to scope visibility.

**Where it's stored**:

```
rel_type     = 'has_mission'
source_id    = entity_projects.id   (the project)
source_type  = 'entity'
target_id    = entity_missions.id   (the mission)
target_type  = 'task' | 'mission'
valid_to     = NULL
```

**Where it can be violated in code**:

- [app/backend/app/api/missions.py](../../../app/backend/app/api/missions.py) — `create_mission()`. Must take `project_id` as required input and create `has_mission` in the same transaction.
- Subtask creation (also in `missions.py`) — when a subtask is created, the writer must create both `parent_id` AND a `has_mission` rel to the parent's project.
- Any importer or seed script that creates missions.

---

## Rule 3 — activity belongs to exactly 1 mission *(non-system activities only)*

**Statement**: every active row in `entity_activities` where `source != 'system'`
has exactly one active `related_to` relationship targeting an
`entity_missions` row.

**Why audit events are excluded**: `capture_activity()` hooks in the
backend create activities for events like "Project created", "Person deleted",
"Organisation created". These exist for the audit trail and don't represent
work attached to a mission. Excluding `source='system'` keeps Rule 3 about
real work.

**Where it's stored**:

```
rel_type     = 'related_to'        (also accepts legacy 'relates_to')
source_id    = entity_activities.id
source_type  = 'activity'
target_id    = entity_missions.id
target_type  = 'task' | 'mission'
valid_to     = NULL
```

**Where it can be violated in code**:

- [app/backend/app/api/activity.py](../../../app/backend/app/api/activity.py) — every activity-create endpoint. Non-system activities must take a `mission_id` as required input.
- [app/backend/app/services/activity_capture.py](../../../app/backend/app/services/activity_capture.py) — `capture_activity()`. This is intentionally for `source='system'` events; it must NEVER write `source='manual'` activities, otherwise they'll bypass Rule 3 and end up orphaned.
- Importers under `app/backend/scripts/`. Past bug: `relates_to` (typo) instead of `related_to` — the validator now treats them as synonyms, but new code must write the canonical name.

---

## Rule 4 — time_entry linked to exactly 1 activity

**Statement**: every active row in `time_entries` has exactly one active
`logged_activity` relationship targeting an `entity_activities` row.

**Why**: a time entry without an activity floats — there's no human-readable
context for what work was done. The chain time_entry → activity → mission
makes the work auditable.

**Where it's stored**:

```
rel_type     = 'logged_activity'   (also accepts legacy 'logged_for')
source_id    = time_entries.id
source_type  = 'time_entry'
target_id    = entity_activities.id
target_type  = 'entity'            (legacy used 'activity'; canonical is 'entity')
valid_to     = NULL
```

**Where it can be violated in code**:

- [app/backend/app/api/time_entries.py](../../../app/backend/app/api/time_entries.py) — `create_time_entry()` and `update_time_entry()`. Must require `activity_id`. Must write the `logged_activity` edge in the same transaction. Must enforce one-active-per-time-entry (uniqueness on `valid_to IS NULL`).
- [app/backend/scripts/import_timelog_from_excel.py](../../../app/backend/scripts/import_timelog_from_excel.py) — past bug: wrote `logged_for` instead of `logged_activity` and `target_type='activity'` instead of `'entity'`. Validator accepts both as synonyms; new writes must use the canonical names.

---

## Rule 5 — time_entry's `logged_against` mission equals its activity's home mission

**Statement**: every active row in `time_entries` has exactly one active
`logged_against` relationship to a mission, **and** that mission must be
identical to the home mission resolved through the time entry's activity
chain (`time_entry → logged_activity → activity → activity-home-link → mission`).
If multiple activities are linked to a single time entry, they must all
share the same home mission.

**Why**: `logged_against` is a server-derived denormalisation of the
activity's home — kept on `time_entries` so effort aggregation, the
project-permission filter, and existing reports keep working without an
extra join. The denormalisation only earns its keep if it's *consistent*
with the activity chain. Plan 065 (activity-anchored time logging)
requires this invariant: the API derives the mission from the activity
on every write, and Rule 5 catches drift, manual edits, or stale rows
that the derivation didn't reach.

**Why not "no direct edge at all"**: an earlier draft of this rule
forbade `time_entry → mission` direct edges entirely. That contradicts
plan 065's target model, which keeps the direct edge as a derived
denormalisation. We removed the anti-rule version and replaced it with
the consistency check.

**Failure modes the validator surfaces**:

| reason | meaning |
|---|---|
| `missing_logged_against` | Time entry has zero active `logged_against` edges |
| `multiple_logged_against` | Time entry has more than one active `logged_against` edge |
| `no_activity_to_verify` | Time entry has no activity link at all (also a Rule 4 violation) |
| `activity_has_no_home` | Linked activity has no home mission edge (also a Rule 3 violation) |
| `activities_span_multiple_missions` | Multiple linked activities resolve to different home missions |
| `logged_against_does_not_match_home` | Direct edge points to a mission different from the derived one |

**Where it's stored**: the canonical write path is `time_entries.py` —
on every create/update of a time entry, the API derives the mission
from the supplied `activity_ids`, then writes a single `logged_against`
edge with that target.

**Where it can be violated in code**:

- [app/backend/app/api/time_entries.py](../../../app/backend/app/api/time_entries.py) — `create_time_entry()`, `update_time_entry()`. Plan 065 introduces `_derive_mission_for_activities()`; that helper must be the *only* writer of `logged_against` rows for time entries. Any code that writes `logged_against` independently of the helper is a candidate violator.
- [app/backend/app/api/missions.py](../../../app/backend/app/api/missions.py) — when an activity's home is changed (re-home), every time entry whose activity is the one that moved must have its `logged_against` updated. Missing this leaves stale direct edges pointing at the old mission.
- [app/backend/app/api/relationships.py](../../../app/backend/app/api/relationships.py) — the generic `POST /api/relationships` endpoint is the loophole. It currently accepts any rel_type without master-data validation; a client could write `logged_against` directly. Plan 065 R13 documents this as a residual risk and recommends rejecting `rel_type='logged_against'` from this endpoint outright.
- Any importer that constructs time entries. They must use the same derivation: pick the activity, look up its `belongs_to` mission (post-065) or `related_to` mission (pre-065), write `logged_against` to that mission.

---

## Adding a new rule

1. Add a `ruleN_*` function in [scripts/validate.py](scripts/validate.py)
   following the same shape: read the snapshot, compute violators, print a
   section, return a finding dict.
2. Add the rule's `rule_N` invocation to the `findings = [...]` list in
   `main()`.
3. Add a section to this file describing the rule, where it's stored, and
   where it can be violated in code.
4. Update the SKILL.md description if the new rule introduces a new term
   the user might mention.
