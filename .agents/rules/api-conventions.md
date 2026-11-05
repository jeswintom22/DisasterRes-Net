# API Conventions

## General

- All endpoints live under `/api`
- Health check at `/health`
- RESTful resource naming
- Feature branches off main

## Backend Patterns

- Async endpoints with FastAPI
- Pydantic models for request/response validation
- SQLAlchemy 2.0 async sessions
- Alembic for all schema migrations

## Frontend-Backend Contract

- Frontend runs on `http://localhost:5173` and proxies `/api` to backend port 8000
- Backend runs on `http://localhost:8000`
- JWT auth with httpOnly cookies

## Domain Model — 5 Primitives

Everything in Loka is built from 5 primitives. All other concepts are relationships between them:

1. **Entity** — people, teams, departments, orgs
2. **Role** — what an entity can do
3. **Task** — units of work (missions)
4. **Event** — things that happened
5. **Artifact** — documents, deliverables, outputs
