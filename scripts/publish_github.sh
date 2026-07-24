#!/usr/bin/env bash
set -euo pipefail

REMOTE_URL="${1:-https://github.com/PanasheC/MA-TRM.git}"
BRANCH="${2:-main}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init -b "$BRANCH"
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Initial MA-TRM reproducible research codebase"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git push -u origin "$BRANCH"
