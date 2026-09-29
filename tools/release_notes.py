"""Print one version's changelog section for a GitHub release."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REFERENCES = re.compile(r"^\[[^\]\n]+\]:[^\n]+$", re.MULTILINE)


def release_notes(changelog: str, version: str) -> str:
    """Extract one nonempty version section and preserve its Markdown links."""
    pattern = rf"^## \[{re.escape(version)}\](?: - [^\n]+)?\n(.*?)(?=^## |\Z)"
    sections = re.findall(pattern, changelog, flags=re.MULTILINE | re.DOTALL)
    if len(sections) != 1:
        raise ValueError(f"Expected exactly one changelog section for {version}; found {len(sections)}.")
    body = REFERENCES.sub("", sections[0]).strip()
    if not body:
        raise ValueError(f"Empty changelog section for {version}.")
    # Repository-relative documentation links need a tag URL on GitHub Releases.
    body = re.sub(
        r"\]\((docs/[^)]+)\)",
        lambda match: f"](https://github.com/quchip/quchip/blob/v{version}/{match[1]})",
        body,
    )
    references = REFERENCES.findall(changelog)
    if references:
        body += "\n\n" + "\n".join(references)
    return body + "\n"


def main() -> None:
    """Read the changelog and fail before publication if the section is invalid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version without the v prefix")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()
    try:
        notes = release_notes(args.changelog.read_text(encoding="utf-8"), args.version)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(notes, end="")


if __name__ == "__main__":
    main()
