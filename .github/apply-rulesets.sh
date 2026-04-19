#!/usr/bin/env bash
# apply-rulesets.sh — sync branch/tag protection rulesets to GitHub via the gh CLI.
#
# Usage:
#   .github/apply-rulesets.sh
#
# Prerequisites:
#   - gh CLI installed and authenticated (gh auth login)
#   - jq installed
#
# The script reads each *-protection.json file in .github/, strips the
# read-only fields (id, source_type, source), and PATCHes the corresponding
# ruleset via the GitHub API. Running it twice is safe — PATCH is idempotent.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

require_cmd() {
  if ! command -v "$1" &>/dev/null; then
    echo "ERROR: '$1' is required but not installed." >&2
    exit 1
  fi
}

require_cmd gh
require_cmd jq

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Apply a single ruleset file
# ---------------------------------------------------------------------------

apply_ruleset() {
  local file="$1"
  local id name

  id=$(jq -r '.id' "$file")
  name=$(jq -r '.name' "$file")

  printf 'Applying %-30s (id: %s) ... ' "$name" "$id"

  # Strip read-only fields before sending to the API.
  # The Rulesets API uses PUT (full replacement), not PATCH.
  jq 'del(.id, .source_type, .source)' "$file" \
    | gh api \
        --method PUT \
        --header "Accept: application/vnd.github+json" \
        --header "X-GitHub-Api-Version: 2022-11-28" \
        "/repos/$REPO/rulesets/$id" \
        --input - \
        --silent

  echo "done"
}

# ---------------------------------------------------------------------------
# Main — apply every *-protection.json in .github/
# ---------------------------------------------------------------------------

echo "Syncing rulesets for $REPO"
echo "-----------------------------------"

shopt -s nullglob
FILES=("$SCRIPT_DIR"/*-protection.json)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No *-protection.json files found in $SCRIPT_DIR" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  apply_ruleset "$f"
done

echo "-----------------------------------"
echo "All rulesets applied."
