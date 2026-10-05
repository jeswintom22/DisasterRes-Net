# Directory Guide

This repository is grouped by purpose. Keep new files in the closest matching
area so the project root remains limited to entry points and configuration.

| Area | Location | Contents |
| --- | --- | --- |
| Application entry points | repository root | `app.py`, `run_all.py`, and the numbered pipeline steps |
| Core ML code | `preprocessing/`, `features/`, `fusion/`, `models/`, `damage_assessment/`, `evaluation/` | Reusable analysis and model modules |
| AI-response service | `crew/` | Crew orchestration, sources, tools, vision, and reporting |
| Web UI | `templates/`, `static/` | Flask HTML templates and stylesheets |
| Supporting code | `config/`, `reports/`, `utils/`, `visualization/`, `tests/` | Configuration, report generation, utilities, and tests |
| Data | `raw_dataset/`, `clean_dataset/`, `split_dataset/` | Source, cleaned, and train/test image data |
| Run artifacts | `results/`, `saved_models/`, `mlruns/`, `mlflow.db` | Generated outputs, trained models, and MLflow tracking data |
| Documentation | `docs/` | Research notes, delivery records, and implementation progress |

## Documentation layout

- `docs/research/`: architecture, validation, audit, status, and final-report documents.
- `docs/project-delivery/`: feature specifications, brainstorms, learnings, and efficiency notes.
- `docs/implementation-progress.md`: CrewAI chat implementation notes.

Runtime folders such as `.venv/`, `.cache/`, `__pycache__/`, and
`.pytest_cache/` are intentionally not part of the source layout.
