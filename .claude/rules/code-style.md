# Code Style Rules

## Backend (Python)

- **Naming**: snake_case for everything
- **Endpoints**: Always async (`async def`)
- **Models**: Pydantic for all request/response schemas
- **Line length**: 100 characters max
- **Linter**: Ruff (E/F/I/W rules, Python 3.11 target)
- **Formatting**: Ruff format

## Frontend (TypeScript/React)

- **Components**: PascalCase
- **Functions/variables**: camelCase
- **CSS custom properties**: Unprefixed tokens (`--accent`, `--bg-primary`). Legacy `--ck-*` aliases exist for backward compatibility.
- **Line length**: 100 characters max
- **Linter**: ESLint + Prettier (single quotes, no semicolons, trailing commas)

## Avatars

Always use `TeamAvatar` (`src/ui/components/shared/TeamAvatar.tsx`) for person/team member avatars.
Never use initials circles, colored dots, or generic User icons for people.
`TeamAvatar` supports sizes `xs | sm | md | lg` and falls back to DiceBear when no `avatarUrl` is provided.

## UI Components

Use shadcn/ui components (`@/components/ui/*`) for all standard UI elements:
- **Buttons**: `<Button>` with variants (`default`, `destructive`, `ghost-muted`, `surface`, `link`), never inline `<button>` with custom Tailwind
- **Inputs**: `<Input>` from `@/components/ui/input`, never inline `<input>` with `bg-elevated border border-border`
- **Textareas**: `<Textarea>` from `@/components/ui/textarea`
- **Badges**: `<Badge>` for status tags and labels
- **Selects**: `<Select>` for dropdowns

Use shared components (`src/ui/components/shared/*`) for cross-page patterns:
- `SearchInput` for search fields, `EmptyState` for empty lists, `FilterPills` for filter toggles
- `ProgressBar` for progress indicators, `SectionHeader` for section headers, `ListItem` for sidebar items

See `context/frontend-architecture.md` for the full component hierarchy and variant guide.

## Frontend Structure

The `src/ui/` directory is organized into three layers:
- `pages/` — one file per route/section, organized by role (`auth/`, `user/`, `admin/`)
- `layouts/` — shell components (AppShell, IconBar, sidebars)
- `components/` — domain components grouped by feature (`chat/`, `missions/`, `admin/`, etc.) with `shared/` for cross-page primitives

Import direction: `pages → layouts → components`. Never import from `pages/` in `components/` or `layouts/`.

## Datetimes

- Database uses `TIMESTAMP WITHOUT TIME ZONE` columns
- **Always use `datetime.utcnow()`** (naive UTC) for DB model fields
- **Never use `datetime.now(timezone.utc)`** for DB writes — asyncpg crashes with offset-naive/offset-aware mismatch
- Reserve `datetime.now(timezone.utc)` only for non-DB contexts (JWT tokens, `.isoformat()` strings)
