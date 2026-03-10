from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r"^(?:revert: )?(?:build|chore|ci|docs|feat|fix|perf|refactor|style|test)"
    r"(?:\([a-z0-9._/-]+\))?(?:!)?: .+"
)

ALLOWED_PREFIXES = (
    "Merge ",
    "Revert ",
    "fixup! ",
    "squash! ",
)


def validate_commit_message(message: str) -> tuple[bool, str]:
    summary = message.strip().splitlines()[0] if message.strip() else ""
    if not summary:
        return False, "Commit message must not be empty."

    if summary.startswith(ALLOWED_PREFIXES):
        return True, ""

    if not CONVENTIONAL_COMMIT_PATTERN.match(summary):
        return (
            False,
            "Invalid commit message. Use Conventional Commits, for example: "
            "'feat(ci): add pre-commit checks' or 'fix: handle empty labels'.",
        )

    if len(summary) > 72:
        return False, "Commit summary must be 72 characters or fewer."

    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate commit messages.")
    parser.add_argument("commit_msg_file", nargs="?", help="Path to the commit message file.")
    parser.add_argument("--message", help="Commit message text to validate directly.")
    args = parser.parse_args()

    if args.message:
        message = args.message
    elif args.commit_msg_file:
        message = Path(args.commit_msg_file).read_text(encoding="utf-8")
    else:
        parser.error("Provide a commit message file or --message.")

    valid, error = validate_commit_message(message)
    if valid:
        return 0

    print(error, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
