#!/usr/bin/env bash
# Fail closed if the compared commits cannot be read.
set -euo pipefail

if [[ "${EVENT_NAME:?}" != "pull_request" ]]; then
  echo "full_ci=true"
  exit 0
fi

changed_files=$(mktemp)
trap 'rm -f "$changed_files"' EXIT
git diff --no-renames --name-only -z "${BASE_SHA:?}" "${HEAD_SHA:?}" > "$changed_files"

full_ci=false
while IFS= read -r -d '' changed_file; do
  case "$changed_file" in
    quchip/__init__.py)
      # A version-only edit changes release metadata, not runtime behavior.
      if ! python3 - "$BASE_SHA" "$HEAD_SHA" <<'PYTHON'
import re
import subprocess
import sys

versions = [subprocess.check_output(["git", "show", f"{ref}:quchip/__init__.py"]) for ref in sys.argv[1:]]
pattern = rb'^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"$'
normalized = [re.subn(pattern, b'__version__ = "VERSION"', source, flags=re.M) for source in versions]
sys.exit(0 if normalized[0] == normalized[1] and normalized[0][1] == 1 else 1)
PYTHON
      then
        full_ci=true
        break
      fi
      ;;
    .github/workflows/*|.github/actions/classify-pr/*)
      ;;
    examples/*|tests/*|quchip/*|tools/*|.github/actions/*|.github/rulesets/*)
      full_ci=true
      break
      ;;
    docs/*|*.md|CITATION.cff|LICENSE|.gitignore|.github/workflows/release.yml)
      ;;
    *)
      full_ci=true
      break
      ;;
  esac
done < "$changed_files"

echo "full_ci=$full_ci"
