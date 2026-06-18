#!/bin/bash
set -euo pipefail

# PostToolUse hook: auto-format frontend files after Edit/Write.
# Receives JSON on stdin with tool_input.file_path.

FILE_PATH=$(cat | jq -r '.tool_input.file_path // empty')

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Only lint TS/TSX/JS/JSX files under app/frontend/src
case "$FILE_PATH" in
  */app/frontend/src/*.ts|*/app/frontend/src/*.tsx|*/app/frontend/src/*.js|*/app/frontend/src/*.jsx) ;;
  *) exit 0 ;;
esac

# Verify file still exists (may have been a delete operation)
if [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR/app/frontend"

# Auto-fix formatting
npx prettier --write "$FILE_PATH" 2>/dev/null || true

# Auto-fix lint issues
npx eslint --fix "$FILE_PATH" 2>/dev/null || true

exit 0
