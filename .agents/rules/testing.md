# Testing Conventions

## Running Tests

### Backend
```bash
cd app/backend && source venv/bin/activate
ruff check .              # lint
ruff format --check .     # format check
```

### Frontend
```bash
cd app/frontend
npm run lint              # ESLint
npm run format:check      # Prettier check
```

## Pre-commit Checklist

1. Run backend lint (`ruff check . && ruff format --check .`)
2. Run frontend lint (`npm run lint && npm run format:check`)
3. Verify Alembic migration chain is linear (`alembic heads` should show one head)
4. Check for TypeScript errors (`npx tsc --noEmit`)
