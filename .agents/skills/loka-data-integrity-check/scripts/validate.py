"""Offline hierarchy validator for a Loka snapshot from extract.py.

Checks the 5 hierarchy invariants in `rules.md`. Tolerant of synonym
rel_types (e.g. legacy `relates_to`/`logged_for`) — anything in the synonym
group counts as the canonical edge.

Outputs:
    - Human-readable report on stdout (one section per rule).
    - Machine-readable findings written to a JSON file (default
      /tmp/loka_integrity_findings.json) so the AI driving the workflow
      can consume them without re-parsing the human output.

Usage (run from anywhere inside the repo):
    python .claude/skills/loka-data-integrity-check/scripts/validate.py \\
        /tmp/loka_integrity_snapshot.json

    # write findings to a custom path
    python .claude/skills/loka-data-integrity-check/scripts/validate.py \\
        /tmp/loka_integrity_snapshot.json \\
        --findings-out /tmp/findings.json

Exit codes:
    0 — all 5 rules green
    1 — at least one rule has violations
    2 — invocation error (missing snapshot, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Rel-type synonym groups. Imports historically wrote misnamed rels
# (relates_to, logged_for); the validator treats them as the canonical edge
# so old data still passes once renamed. The activity-home synonym set also
# tolerates the post-065 canonical name `belongs_to | source=activity, target
# in (task,mission)` so the validator works on both pre- and post-065 data.
ACTIVITY_HOME_SYNONYMS = {"related_to", "relates_to", "belongs_to"}
RULE3_SYNONYMS = ACTIVITY_HOME_SYNONYMS  # legacy alias
RULE4_SYNONYMS = {"logged_activity", "logged_for"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load(path: Path) -> dict:
    return json.loads(path.read_text())


def active_rels(rels: list[dict]) -> list[dict]:
    return [r for r in rels if r["valid_to"] is None]


def index_by_id(rows: list[dict]) -> dict[str, dict]:
    return {r["id"]: r for r in rows}


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_distribution(label: str, counts: Counter) -> None:
    print(f"  {label}")
    for k in sorted(counts):
        print(f"    {k:>3} → {counts[k]:>5}")


# ---------------------------------------------------------------------------
# Rule implementations — each returns (text-printed, finding-dict)
# ---------------------------------------------------------------------------
def rule1_project_to_org(snap: dict) -> dict:
    section("RULE 1 — project belongs to exactly 1 organisation")
    projects = snap["tables"]["entity_projects"]
    orgs_by_id = index_by_id(snap["tables"]["entity_organisations"])
    rels = active_rels(snap["tables"]["relationships"])

    active_projects = [p for p in projects if p.get("active") and p.get("status") == "active"]
    project_ids = {p["id"] for p in active_projects}
    per_project: dict[str, set[str]] = defaultdict(set)

    # belongs_to in either direction (writers used both shapes historically).
    for r in rels:
        if r["rel_type"] != "belongs_to":
            continue
        src_is_project = r["source_id"] in project_ids
        tgt_is_project = r["target_id"] in project_ids
        if not (src_is_project or tgt_is_project):
            continue
        project_id = r["source_id"] if src_is_project else r["target_id"]
        other_id = r["target_id"] if src_is_project else r["source_id"]
        if other_id in orgs_by_id:
            per_project[project_id].add(other_id)

    counts: Counter = Counter()
    violators = []
    for p in active_projects:
        n = len(per_project.get(p["id"], set()))
        counts[n] += 1
        if n != 1:
            violators.append(
                {"id": p["id"], "name": p["name"], "tenant_id": p["tenant_id"], "count": n}
            )

    print(f"  active projects: {len(active_projects)}")
    print_distribution("org-count distribution:", counts)
    print(f"\n  violators ({len(violators)}):")
    for v in sorted(violators, key=lambda x: (x["count"], x["name"])):
        linked = per_project.get(v["id"], set())
        linked_names = ", ".join(sorted(orgs_by_id[oid]["name"] for oid in linked)) or "<none>"
        print(f"    [{v['count']}] {v['name']:<40} ({v['id']})  → {linked_names}")

    return {
        "rule": "rule1",
        "title": "project → 1 organisation",
        "passed": len(violators) == 0,
        "active_total": len(active_projects),
        "distribution": dict(counts),
        "violators": violators,
    }


def rule2_mission_to_project(snap: dict) -> dict:
    section("RULE 2 — mission belongs to exactly 1 project")
    missions = snap["tables"]["entity_missions"]
    projects_by_id = index_by_id(snap["tables"]["entity_projects"])
    rels = active_rels(snap["tables"]["relationships"])

    mission_ids = {m["id"] for m in missions}
    per_mission: dict[str, set[str]] = defaultdict(set)

    for r in rels:
        if r["rel_type"] != "has_mission":
            continue
        if r["target_id"] in mission_ids and r["source_id"] in projects_by_id:
            per_mission[r["target_id"]].add(r["source_id"])

    counts: Counter = Counter()
    subtask_counts: Counter = Counter()
    violators = []
    for m in missions:
        n = len(per_mission.get(m["id"], set()))
        counts[n] += 1
        is_subtask = m.get("parent_id") is not None
        subtask_counts[(is_subtask, n)] += 1
        if n != 1:
            violators.append(
                {
                    "id": m["id"],
                    "title": m["title"],
                    "tenant_id": m["tenant_id"],
                    "is_subtask": is_subtask,
                    "count": n,
                }
            )

    print(f"  total missions: {len(missions)}")
    print_distribution("project-count distribution:", counts)
    print("  by (is_subtask, project_count):")
    for (subtask, n), c in sorted(subtask_counts.items()):
        tag = "subtask  " if subtask else "top-level"
        print(f"    {tag} count={n:>2} → {c}")
    print(f"\n  violators ({len(violators)}):")
    for v in sorted(violators, key=lambda x: (x["count"], x["title"])):
        tag = "subtask" if v["is_subtask"] else "top-level"
        print(f"    [{v['count']}] [{tag}] {v['title'][:60]:<60} ({v['id']})")

    return {
        "rule": "rule2",
        "title": "mission → 1 project",
        "passed": len(violators) == 0,
        "active_total": len(missions),
        "distribution": dict(counts),
        "violators": violators,
    }


def rule3_activity_to_mission(snap: dict) -> dict:
    section("RULE 3 — activity belongs to exactly 1 mission (excluding source='system')")
    activities = snap["tables"]["entity_activities"]
    missions_by_id = index_by_id(snap["tables"]["entity_missions"])
    rels = active_rels(snap["tables"]["relationships"])

    # Audit-log activities (project_created, person_deleted, etc. produced by
    # `capture_activity()` hooks) carry source='system'. They're noise from a
    # rule-3 standpoint — there's no work-on-a-mission to attach.
    active_acts = [
        a
        for a in activities
        if a.get("active") and a.get("status") == "active" and a.get("source") != "system"
    ]
    act_ids = {a["id"] for a in active_acts}
    per_act: dict[str, set[str]] = defaultdict(set)

    # Activity home is tracked under any of the synonyms (related_to / relates_to
    # / belongs_to), but for activity sources only count edges with target in
    # missions. Generic activity→entity relates_to edges (e.g. "activity is
    # about person X") are NOT home links and must be excluded.
    for r in rels:
        if r["rel_type"] not in ACTIVITY_HOME_SYNONYMS:
            continue
        if r["source_id"] not in act_ids:
            continue
        if r["target_id"] in missions_by_id:
            per_act[r["source_id"]].add(r["target_id"])

    counts: Counter = Counter()
    violators = []
    for a in active_acts:
        n = len(per_act.get(a["id"], set()))
        counts[n] += 1
        if n != 1:
            violators.append(
                {
                    "id": a["id"],
                    "name": a["name"],
                    "type": a.get("type"),
                    "source": a.get("source"),
                    "tenant_id": a["tenant_id"],
                    "count": n,
                }
            )

    synonym_usage: Counter = Counter()
    for r in rels:
        if (
            r["rel_type"] in ACTIVITY_HOME_SYNONYMS
            and r["source_id"] in act_ids
            and r["target_id"] in missions_by_id
        ):
            synonym_usage[r["rel_type"]] += 1

    print(f"  active activities (non-system): {len(active_acts)}")
    print(f"  synonym usage: {dict(synonym_usage)}")
    print_distribution("mission-count distribution:", counts)
    multi = [v for v in violators if v["count"] > 1]
    zero = [v for v in violators if v["count"] == 0]
    print(f"\n  violators: {len(violators)} (multi={len(multi)}, zero={len(zero)})")
    for v in sorted(multi, key=lambda x: -x["count"])[:20]:
        print(f"    [{v['count']}] {v['name'][:80]:<80} ({v['id']})")
    if zero:
        print(f"    ... and {len(zero)} activities with zero mission links")

    return {
        "rule": "rule3",
        "title": "activity (non-system) → 1 mission",
        "passed": len(violators) == 0,
        "active_total": len(active_acts),
        "distribution": dict(counts),
        "synonym_usage": dict(synonym_usage),
        "violators": violators,
    }


def rule4_time_entry_to_activity(snap: dict) -> dict:
    section("RULE 4 — time_entry linked to exactly 1 activity")
    time_entries = snap["tables"]["time_entries"]
    activities_by_id = index_by_id(snap["tables"]["entity_activities"])
    rels = active_rels(snap["tables"]["relationships"])

    active_te = [t for t in time_entries if t.get("status") == "active"]
    te_ids = {t["id"] for t in active_te}
    per_te: dict[str, set[str]] = defaultdict(set)

    for r in rels:
        if r["rel_type"] not in RULE4_SYNONYMS:
            continue
        if r["source_id"] not in te_ids:
            continue
        if r["target_id"] in activities_by_id:
            per_te[r["source_id"]].add(r["target_id"])

    counts: Counter = Counter()
    violators = []
    for t in active_te:
        n = len(per_te.get(t["id"], set()))
        counts[n] += 1
        if n != 1:
            violators.append(
                {
                    "id": t["id"],
                    "tenant_id": t["tenant_id"],
                    "date": t.get("date"),
                    "minutes": t.get("minutes"),
                    "count": n,
                }
            )

    synonym_usage: Counter = Counter()
    for r in rels:
        if r["rel_type"] in RULE4_SYNONYMS and r["source_id"] in te_ids:
            synonym_usage[r["rel_type"]] += 1

    print(f"  active time_entries: {len(active_te)}")
    print(f"  synonym usage: {dict(synonym_usage)}")
    print_distribution("activity-count distribution:", counts)

    multi = [v for v in violators if v["count"] > 1]
    zero = [v for v in violators if v["count"] == 0]
    print(f"\n  violators: {len(violators)} (multi={len(multi)}, zero={len(zero)})")
    for v in sorted(multi, key=lambda x: -x["count"])[:20]:
        print(f"    [{v['count']}] te {v['id']:<40} date={v['date']} mins={v['minutes']}")

    return {
        "rule": "rule4",
        "title": "time_entry → 1 activity",
        "passed": len(violators) == 0,
        "active_total": len(active_te),
        "distribution": dict(counts),
        "synonym_usage": dict(synonym_usage),
        "violators": violators,
    }


def rule5_logged_against_matches_activity_home(snap: dict) -> dict:
    section("RULE 5 — time_entry.logged_against mission == activity's home mission")
    rels = active_rels(snap["tables"]["relationships"])
    time_entries = snap["tables"]["time_entries"]
    activities_by_id = index_by_id(snap["tables"]["entity_activities"])
    missions_by_id = index_by_id(snap["tables"]["entity_missions"])

    active_te = [t for t in time_entries if t.get("status") == "active"]
    te_ids = {t["id"] for t in active_te}

    # Index 1: time_entry → set of logged_against mission targets.
    te_logged_against: dict[str, set[str]] = defaultdict(set)
    for r in rels:
        if r["rel_type"] != "logged_against":
            continue
        if r["source_id"] not in te_ids:
            continue
        if r["target_id"] in missions_by_id:
            te_logged_against[r["source_id"]].add(r["target_id"])

    # Index 2: time_entry → set of linked activity ids (via logged_activity synonyms).
    te_activities: dict[str, set[str]] = defaultdict(set)
    for r in rels:
        if r["rel_type"] not in RULE4_SYNONYMS:
            continue
        if r["source_id"] not in te_ids:
            continue
        if r["target_id"] in activities_by_id:
            te_activities[r["source_id"]].add(r["target_id"])

    # Index 3: activity → set of home mission ids (via activity-home synonyms,
    # only edges that target a real mission row).
    act_home: dict[str, set[str]] = defaultdict(set)
    for r in rels:
        if r["rel_type"] not in ACTIVITY_HOME_SYNONYMS:
            continue
        if r["source_id"] not in activities_by_id:
            continue
        if r["target_id"] in missions_by_id:
            act_home[r["source_id"]].add(r["target_id"])

    # For each time_entry, derive the expected mission set from its activities
    # and compare against logged_against.
    violators = []
    fail_reasons: Counter = Counter()
    for t in active_te:
        te_id = t["id"]
        direct = te_logged_against.get(te_id, set())
        activity_ids = te_activities.get(te_id, set())
        derived = set()
        for aid in activity_ids:
            derived |= act_home.get(aid, set())

        # Failure modes
        if len(direct) == 0:
            fail_reasons["missing_logged_against"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "missing_logged_against",
                    "direct_missions": [],
                    "derived_missions": sorted(derived),
                    "activity_count": len(activity_ids),
                }
            )
            continue
        if len(direct) > 1:
            fail_reasons["multiple_logged_against"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "multiple_logged_against",
                    "direct_missions": sorted(direct),
                    "derived_missions": sorted(derived),
                    "activity_count": len(activity_ids),
                }
            )
            continue
        if len(activity_ids) == 0:
            # Already a Rule 4 violator; report here too because we cannot
            # verify the consistency without an activity to derive from.
            fail_reasons["no_activity_to_verify"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "no_activity_to_verify",
                    "direct_missions": sorted(direct),
                    "derived_missions": [],
                    "activity_count": 0,
                }
            )
            continue
        if len(derived) == 0:
            fail_reasons["activity_has_no_home"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "activity_has_no_home",
                    "direct_missions": sorted(direct),
                    "derived_missions": [],
                    "activity_count": len(activity_ids),
                }
            )
            continue
        if len(derived) > 1:
            fail_reasons["activities_span_multiple_missions"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "activities_span_multiple_missions",
                    "direct_missions": sorted(direct),
                    "derived_missions": sorted(derived),
                    "activity_count": len(activity_ids),
                }
            )
            continue
        if direct != derived:
            fail_reasons["logged_against_does_not_match_home"] += 1
            violators.append(
                {
                    "time_entry_id": te_id,
                    "tenant_id": t["tenant_id"],
                    "reason": "logged_against_does_not_match_home",
                    "direct_missions": sorted(direct),
                    "derived_missions": sorted(derived),
                    "activity_count": len(activity_ids),
                }
            )

    print(f"  active time_entries checked: {len(active_te)}")
    print(f"  violators: {len(violators)}")
    if fail_reasons:
        print("  by reason:")
        for reason, n in fail_reasons.most_common():
            print(f"    {reason:<40} → {n}")
    if violators:
        print("  sample (first 10):")
        for v in violators[:10]:
            print(
                f"    [{v['reason']}] te={v['time_entry_id']} "
                f"direct={v['direct_missions']} derived={v['derived_missions']}"
            )

    return {
        "rule": "rule5",
        "title": "time_entry.logged_against == activity's home mission",
        "passed": len(violators) == 0,
        "active_total": len(active_te),
        "violators": violators,
        "fail_reasons": dict(fail_reasons),
    }


def edge_shape_audit(snap: dict) -> None:
    section("EDGE-SHAPE AUDIT — (rel_type, source_type, target_type) counts")
    rels = active_rels(snap["tables"]["relationships"])
    shapes: Counter = Counter()
    for r in rels:
        shapes[(r["rel_type"], r["source_type"], r["target_type"])] += 1

    interesting = (
        "belongs_to",
        "has_mission",
        "related_to",
        "relates_to",
        "logged_activity",
        "logged_for",
        "logged_against",
    )
    for rt in interesting:
        subset = {k: v for k, v in shapes.items() if k[0] == rt}
        if not subset:
            continue
        print(f"  {rt}")
        for (_, st, tt), n in sorted(subset.items(), key=lambda kv: -kv[1]):
            print(f"    source_type={st:<12} target_type={tt:<14} → {n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", type=Path, help="path to JSON snapshot from extract.py")
    ap.add_argument(
        "--findings-out",
        type=Path,
        default=Path("/tmp/loka_integrity_findings.json"),
        help="where to write structured findings (default: /tmp/loka_integrity_findings.json)",
    )
    args = ap.parse_args()

    if not args.snapshot.exists():
        print(f"snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(2)

    snap = load(args.snapshot)
    print(f"snapshot exported_at: {snap.get('exported_at')}  env={snap.get('env')}")
    for table, rows in snap["tables"].items():
        print(f"  {table:<22} {len(rows):>7} rows")

    edge_shape_audit(snap)
    findings = [
        rule1_project_to_org(snap),
        rule2_mission_to_project(snap),
        rule3_activity_to_mission(snap),
        rule4_time_entry_to_activity(snap),
        rule5_logged_against_matches_activity_home(snap),
    ]

    report = {
        "snapshot_exported_at": snap.get("exported_at"),
        "snapshot_env": snap.get("env"),
        "rules": findings,
        "all_passed": all(f["passed"] for f in findings),
    }
    args.findings_out.write_text(json.dumps(report, indent=2, default=str))
    print()
    print(f"Findings written to {args.findings_out}")
    print()
    print("Summary:")
    for f in findings:
        status = "✓ PASS" if f["passed"] else "✗ FAIL"
        print(f"  {status}  {f['rule']}  {f['title']}")

    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
